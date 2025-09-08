#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace fused_local_corr {

__global__ void fused_local_corr_1d_nearest_cuda(int B, int HW, int N, int C, int H, int W, const float* __restrict__ im_A, const float* __restrict__ im_B, const int* __restrict__ warp, float* __restrict__ result) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B*HW*N){
    return;
  }
  int b = t/(HW*N);
  int hwn = t%(HW*N);
  int hw = hwn/N;
  int n = hwn % N;
  int D = 2;

  // Lookup indices
  int w = b*(HW*N*D) + hw*(N*D) + n*D;
  int x = warp[w];
  int y = warp[w+1];
  int ia = b*HW*C + hw*C; // index for im_A
  int ib = b*H*W*C + y*W*C + x*C; // index for im_B
  int r  = b*HW*N + hw*N + n; // index for result

  // Accumulate dot product in registers and write once
  float sum = 0.0f;
  int c = 0;
  // Vectorized path when possible
  for (; c + 3 < C; c += 4) {
    const float4 a = *reinterpret_cast<const float4*>(&im_A[ia + c]);
    const float4 b4 = *reinterpret_cast<const float4*>(&im_B[ib + c]);
    sum += a.x * b4.x + a.y * b4.y + a.z * b4.z + a.w * b4.w;
  }
  // Tail
  for (; c < C; ++c) {
    sum += im_A[ia + c] * im_B[ib + c];
  }
  result[r] = sum;
}

__global__ void fused_local_corr_1d_bilinear_cuda(int B, int HW, int N, int C, int H, int W, const float* im_A, const float* im_B, const float* warp, float* result) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B*HW*N){
    return;
  }
  int b = t/(HW*N);
  int hwn = t%(HW*N);
  int hw = hwn/N;
  int n = hwn % N;
  int D = 2;
  int w = b*(HW*N*D) + hw*(N*D) + n*D;
  float x = warp[w]-0.5;
  int x_low = std::floor(x);
  float x_frac = x-x_low;
  float y = warp[w+1]-0.5;
  int y_low = std::floor(y);
  float y_frac = y-y_low;
  int ia = b*HW*C + hw*C; //index for im_A
  int ib = b*H*W*C + y_low*W*C + x_low*C; //index for im_B
  int r = b*HW*N + hw*N + n; //index for result
  // Unrolled and optimized version of the loop
  float alpha_y_0 = (1.0f-y_frac);  // Precompute constants
  float alpha_y_1 = y_frac;
  float alpha_x_0 = (1.0f-x_frac);
  float alpha_x_1 = x_frac;

  float sum = 0.0f;

  // y = 0 case
  if (y_low >= 0 && y_low < H) {
      // x = 0 case
      if (x_low >= 0 && x_low < W) {
          float alpha = alpha_y_0 * alpha_x_0;
          #pragma unroll
          for (int c = 0; c < C; c += 4) {
              const float4 a = *reinterpret_cast<const float4*>(&im_A[ia+c]);
              const float4 b = *reinterpret_cast<const float4*>(&im_B[ib+c]);
              sum += alpha * (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
          }
      }
      
      // x = 1 case
      if (x_low + 1 >= 0 && x_low + 1 < W) {
          float alpha = alpha_y_0 * alpha_x_1;
          #pragma unroll
          for (int c = 0; c < C; c += 4) {
              const float4 a = *reinterpret_cast<const float4*>(&im_A[ia+c]);
              const float4 b = *reinterpret_cast<const float4*>(&im_B[ib+C+c]);
              sum += alpha * (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
          }
      }
  }

  // y = 1 case
  if (y_low + 1 >= 0 && y_low + 1 < H) {
      int ib_offset = ib + W*C;
      
      // x = 0 case
      if (x_low >= 0 && x_low < W) {
          float alpha = alpha_y_1 * alpha_x_0;
          #pragma unroll
          for (int c = 0; c < C; c += 4) {
              const float4 a = *reinterpret_cast<const float4*>(&im_A[ia+c]);
              const float4 b = *reinterpret_cast<const float4*>(&im_B[ib_offset+c]);
              sum += alpha * (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
          }
      }
      
      // x = 1 case
      if (x_low + 1 >= 0 && x_low + 1 < W) {
          float alpha = alpha_y_1 * alpha_x_1;
          #pragma unroll
          for (int c = 0; c < C; c += 4) {
              const float4 a = *reinterpret_cast<const float4*>(&im_A[ia+c]);
              const float4 b = *reinterpret_cast<const float4*>(&im_B[ib_offset+C+c]);
              sum += alpha * (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
          }
      }
  }

  result[r] = sum;
}

__global__ void fused_local_corr_1d_backward_wrt_A_nearest_cuda(int B, int pixels_A, int N, int C, int H, int W, const float* grad, const float* im_B, const int* warp, float* grad_A) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B*pixels_A){
    return;
  }
  int b = t/pixels_A;
  int hw = t%pixels_A;
  int D = 2;
  // for whatever hypothesis
  for(int n=0; n < N; ++n){
    // Now we know the index of the warp and thus of B (and implictly A)
    int w = b*(pixels_A*N*D) + hw*(N*D) + n*D;
    int x = warp[w];
    int y = warp[w+1];
    int ia = b*pixels_A*C + hw*C; //index for im_A
    int ib = b*H*W*C + y*W*C + x*C; //index for im_B
    int r = b*pixels_A*N + hw*N + n; //index for result
    for(int c=0; c < C; ++c){
      grad_A[ia+c] += grad[r]*im_B[ib+c];
    }
  }
}

__global__ void fused_local_corr_1d_backward_wrt_A_bilinear_cuda(int B, int HW_A, int N, int C, int H, int W, const float* grad, const float* im_B, const float* warp, float* grad_A) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B*HW_A*C){
    return;
  }
  int b = t/(HW_A*C);
  int hwc = t%(HW_A*C);
  int hw = hwc/C;
  int c = hwc % C;
  int D = 2;
  float sum = 0;
  for(int n=0; n < N; ++n){
    int w = b*(HW_A*N*D) + hw*(N*D) + n*D;
    float x = warp[w]-0.5;
    int x_low = std::floor(x);
    float x_frac = x-x_low;
    float y = warp[w+1]-0.5;
    int y_low = std::floor(y);
    float y_frac = y-y_low;
    int ib = b*H*W*C + y_low*W*C + x_low*C; //index for im_B
    int r = b*HW_A*N + hw*N + n; //index for result
    for(int off_y=0; off_y < 2; off_y++){
      if ((y_low + off_y) > H-1 || (y_low + off_y) < 0){
        continue;
      }
      float alpha_y = (1-off_y)*(1-y_frac) + off_y*y_frac;
      for(int off_x=0; off_x < 2; off_x++){
        if ((x_low + off_x) > W-1 || (x_low + off_x) < 0){
          continue;
        }
        float alpha_x = (1-off_x)*(1-x_frac) + off_x*x_frac;
        int off_ib = ib + off_y * W*C + off_x * C;
        float alpha = alpha_y*alpha_x;
        sum += alpha*grad[r]*im_B[off_ib+c];
      }
    }
  }
  grad_A[t] = sum;
}

at::Tensor corr_cuda(const at::Tensor& im_A, const at::Tensor& im_B, const at::Tensor& warp, const std::string mode = "nearest") {
  // TORCH_CHECK(im_A.dtype() == at::kFloat);
  // TORCH_CHECK(im_B.dtype() == at::kFloat);
  // TORCH_CHECK(warp.dtype() == at::kInt);

  TORCH_CHECK(im_A.dim() == 3);
  TORCH_CHECK(im_B.dim() == 4);
  TORCH_CHECK(warp.dim() == 4);

  // Check same batch
  TORCH_CHECK(im_A.size(0) == im_B.size(0) && im_B.size(0) == warp.size(0));
  
  // Check same hw
  TORCH_CHECK(im_A.size(1) == warp.size(1));
  
  // Check same channels
  TORCH_CHECK(im_A.size(2) == im_B.size(3));

  TORCH_INTERNAL_ASSERT(im_A.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(im_B.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(warp.device().type() == at::DeviceType::CUDA);

  at::Tensor im_A_contig = im_A.contiguous();
  at::Tensor im_B_contig = im_B.contiguous();
  at::Tensor warp_contig = warp.contiguous();

  int B = im_A_contig.size(0);
  int HW = im_A_contig.size(1);
  int H = im_B_contig.size(1);
  int W = im_B_contig.size(2);
  int C = im_B_contig.size(3);
  int N = warp_contig.size(2);


  at::Tensor result = torch::zeros({B, HW, N}, im_A_contig.options());
  int numel = B*HW*N;
  const float* im_A_ptr = im_A_contig.to(torch::kFloat).data_ptr<float>();
  const float* im_B_ptr = im_B_contig.to(torch::kFloat).data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  if(mode == "nearest"){
    const auto warp_i = warp_contig.to(torch::kInt);
    const int* warp_ptr = warp_i.data_ptr<int>();
    fused_local_corr_1d_nearest_cuda<<<(numel+255)/256, 256>>>(B, HW, N, C, H, W, im_A_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  else if(mode == "bilinear"){
    const auto warp_f = warp_contig.to(torch::kFloat);
    const float* warp_ptr = warp_f.data_ptr<float>();
    fused_local_corr_1d_bilinear_cuda<<<(numel+255)/256, 256>>>(B, HW, N, C, H, W, im_A_ptr, im_B_ptr, warp_ptr, result_ptr);
  }

  return result;
}

at::Tensor corr_cuda_backward_wrt_A(const at::Tensor& grad, const at::Tensor& im_B, const at::Tensor& warp, const std::string mode = "nearest") {
  // grad.shape = B, pixels_A, N  
  // b.shape = B, H, W,  
  // c.shape = B, pixels_A, N, D (D=2)
  // result.shape = B, HW, C 
  TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(im_B.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(warp.device().type() == at::DeviceType::CUDA);


  int B = im_B.size(0);
  int HW_A = warp.size(1);
  int H = im_B.size(1);
  int W = im_B.size(2);
  int C = im_B.size(3);
  int N = warp.size(2);
  const auto im_B_contig = im_B.contiguous();
  const auto warp_contig = warp.contiguous();
  const auto grad_contig = grad.contiguous();


  at::Tensor result = torch::zeros({B, HW_A, C}, im_B_contig.options());
  const float* grad_ptr = grad_contig.to(torch::kFloat).data_ptr<float>();
  const float* im_B_ptr = im_B_contig.to(torch::kFloat).data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = B*HW_A*C;
  if(mode == "nearest"){
    const auto shit = warp_contig.to(torch::kInt);
    const int* warp_ptr = shit.data_ptr<int>();
    fused_local_corr_1d_backward_wrt_A_nearest_cuda<<<(numel+255)/256, 256>>>(B, HW_A, N, C, H, W, grad_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  else if(mode == "bilinear"){
    const auto shit = warp_contig.to(torch::kFloat);
    const float* warp_ptr = shit.data_ptr<float>();
    fused_local_corr_1d_backward_wrt_A_bilinear_cuda<<<(numel+255)/256, 256>>>(B, HW_A, N, C, H, W, grad_ptr, im_B_ptr, warp_ptr, result_ptr);
  }  
  return result;
}

// Registers CUDA implementations for corr
TORCH_LIBRARY_IMPL(fused_local_corr, CUDA, m) {
  m.impl("corr", &corr_cuda);
  m.impl("corr_backward_A", &corr_cuda_backward_wrt_A);
}
}