#include <torch/extension.h>
#include <omp.h>
#include <vector>
#include <cmath>

namespace fused_local_corr {


void fused_local_corr_1d_nearest(int B, int HW, int N, int C, int H, int W, const float* im_A, const float* im_B, const int* warp, float* result) {
  int D = 2;
  // for sample in batch
  for(int b = 0; b < B; ++b){
    // for whatever pixels in A
    #pragma omp parallel for schedule(static)
    for(int hw = 0; hw < HW; ++hw){
      // for whatever hypothesis
      for(int n=0; n < N; ++n){
        // Now we know the index of the warp and thus of B (and implictly A)
        int w = b*(HW*N*D) + hw*(N*D) + n*D;
        int x = warp[w];
        int y = warp[w+1];
        int ia = b*HW*C + hw*C; //index for im_A
        int ib = b*H*W*C + y*W*C + x*C; //index for im_B
        int r = b*HW*N + hw*N + n; //index for result
        // Next just iterate over the channels to get the correlation
        for(int c=0; c < C; ++c){
          result[r] += im_A[ia+c]*im_B[ib+c];
        }
      }
    }
  }
}

void fused_local_corr_1d_bilinear(int B, int HW, int N, int C, int H, int W, const float* im_A, const float* im_B, const float* warp, float* result) {
  int D = 2;
  // for sample in batch
  for(int b = 0; b < B; ++b){
    // for whatever pixels in A
    #pragma omp parallel for schedule(static)
    for(int hw = 0; hw < HW; ++hw){
      // for whatever hypothesis
      for(int n=0; n < N; ++n){
        // Now we know the index of the warp and thus of B (and implictly A)
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
            // Next just iterate over the channels to get the correlation
            for(int c=0; c < C; ++c){
              result[r] += alpha*im_A[ia+c]*im_B[off_ib+c];
            }
          }
        }
      }
    }
  }
}


void fused_local_corr_1d_backward_wrt_A_nearest(int B, int pixels_A, int N, int C, int H, int W, const float* grad, const float* im_B, const int* warp, float* grad_A) {
  int D = 2;
  // for sample in batch
  for(int b = 0; b < B; ++b){
    // for whatever pixels in A
    #pragma omp parallel for schedule(static)
    for(int hw = 0; hw < pixels_A; ++hw){
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
  }
}

void fused_local_corr_1d_backward_wrt_A_bilinear(int B, int pixels_A, int N, int C, int H, int W, const float* grad, const float* im_B, const float* warp, float* grad_A) {
  int D = 2;
  // for sample in batch
  for(int b = 0; b < B; ++b){
    // for whatever pixels in A
    #pragma omp parallel for schedule(static)
    for(int hw = 0; hw < pixels_A; ++hw){
      // for whatever hypothesis
      for(int n=0; n < N; ++n){
        // Now we know the index of the warp and thus of B (and implictly A)
        int w = b*(pixels_A*N*D) + hw*(N*D) + n*D;
        float x = warp[w]-0.5;
        int x_low = std::floor(x);
        float x_frac = x-x_low;
        float y = warp[w+1]-0.5;
        int y_low = std::floor(y);
        float y_frac = y-y_low;
        int ia = b*pixels_A*C + hw*C; //index for im_A
        int ib = b*H*W*C + y_low*W*C + x_low*C; //index for im_B
        int r = b*pixels_A*N + hw*N + n; //index for result
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
            // Next just iterate over the channels to get the correlation
            // std::cout << B << pixels_A << H << W << C << N << std::endl;
            for(int c=0; c < C; ++c){
              grad_A[ia+c] += alpha*grad[r]*im_B[off_ib+c];//im_B[0];//alpha*grad[r]*//
            }
          }
        }
      }
    }
  }
}

// An example of an operator that mutates one of its inputs.
at::Tensor corr_cpu(const at::Tensor& im_A, const at::Tensor& im_B, const at::Tensor& warp, const std::string mode = "nearest") {
  // a.shape = B, HW, C 
  // b.shape = B, H, W, C 
  // c.shape = B, HW, N, D (D=2)
  // Check same batch
  TORCH_CHECK(im_A.size(0) == im_B.size(0) && im_B.size(0) == warp.size(0));
  
  // Check same hw
  TORCH_CHECK(im_A.size(1) == warp.size(1));
  
  // Check same channels
  TORCH_CHECK(im_A.size(2) == im_B.size(3));

  TORCH_INTERNAL_ASSERT(im_A.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(im_B.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(warp.device().type() == at::DeviceType::CPU);


  int B = im_A.size(0);
  int HW = im_A.size(1);
  int H = im_B.size(1);
  int W = im_B.size(2);
  int C = im_B.size(3);
  int N = warp.size(2);

  at::Tensor im_A_contig = im_A.contiguous();
  at::Tensor im_B_contig = im_B.contiguous();
  at::Tensor warp_contig = warp.contiguous();

  at::Tensor result = torch::zeros({B, HW, N}, im_A.options());
  const float* im_A_ptr = im_A_contig.to(torch::kFloat).data_ptr<float>();
  const float* im_B_ptr = im_B_contig.to(torch::kFloat).data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  if(mode == "nearest"){
    const auto shit = warp_contig.to(torch::kInt);
    const int* warp_ptr = shit.to(torch::kInt).data_ptr<int>();
    fused_local_corr_1d_nearest(B, HW, N, C, H, W, im_A_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  else if(mode == "bilinear"){
    const auto shit = warp.contiguous().to(torch::kFloat);
    const float* warp_ptr = warp.data_ptr<float>();
    fused_local_corr_1d_bilinear(B, HW, N, C, H, W, im_A_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  return result;
}

at::Tensor corr_cpu_backward_wrt_A(const at::Tensor& grad, const at::Tensor& im_B, const at::Tensor& warp, const std::string mode = "nearest") {
  // grad.shape = B, pixels_A, N  
  // b.shape = B, H, W,  
  // c.shape = B, pixels_A, N, D (D=2)
  // result.shape = B, HW, C 
  TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(im_B.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(warp.device().type() == at::DeviceType::CPU);

  int B = im_B.size(0);
  int pixels_A = warp.size(1);
  int H = im_B.size(1);
  int W = im_B.size(2);
  int C = im_B.size(3);
  int N = warp.size(2);
  const auto im_B_contig = im_B.contiguous();
  const auto warp_contig = warp.contiguous();
  const auto grad_contig = grad.contiguous();


  at::Tensor result = torch::zeros({B, pixels_A, C}, im_B_contig.options());
  const float* grad_ptr = grad_contig.to(torch::kFloat).data_ptr<float>();
  const float* im_B_ptr = im_B_contig.to(torch::kFloat).data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  if(mode == "nearest"){
    const auto shit = warp_contig.to(torch::kInt);
    const int* warp_ptr = shit.data_ptr<int>();
    fused_local_corr_1d_backward_wrt_A_nearest(B, pixels_A, N, C, H, W, grad_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  else if(mode == "bilinear"){
    const auto shit = warp_contig.to(torch::kFloat);
    const float* warp_ptr = shit.data_ptr<float>();
    fused_local_corr_1d_backward_wrt_A_bilinear(B, pixels_A, N, C, H, W, grad_ptr, im_B_ptr, warp_ptr, result_ptr);
  }
  return result;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(fused_local_corr, m) {
  m.def("corr(Tensor im_A, Tensor im_B, Tensor warp, str mode) -> Tensor");
  m.def("corr_backward_A(Tensor grad, Tensor im_B, Tensor warp, str mode) -> Tensor");
}

// Registers CUDA implementations for corr
TORCH_LIBRARY_IMPL(fused_local_corr, CPU, m) {
  m.impl("corr", &corr_cpu);
  m.impl("corr_backward_A", &corr_cpu_backward_wrt_A);
}

}
