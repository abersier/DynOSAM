#include "dynosam/frontend/vision/cuda_backproject.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// GPU-accelerated pinhole backprojection.
// CPU reference: Frame.cc::projectToDenseCloud.
// Mirrors structure of dyno_mppi_critic/cuda_sdf_query.cu.

namespace dyno::cuda {

// ── Kernel ──────────────────────────────────────────────────────────────────
//
// One thread per pixel. Valid pixels are backprojected and written compactly
// via atomicAdd. background_label = 0 (constexpr in DynoSAM Types.hpp).

__global__ void backprojectKernel(
    const float*   __restrict__ d_depth,
    const int32_t* __restrict__ d_mask,
    const uint8_t* __restrict__ d_det_mask,  // nullptr = no detection mask
    int rows, int cols,
    float fx_inv, float fy_inv,
    float cx, float cy,
    float max_bg_depth, float max_obj_depth,
    float*   __restrict__ d_xyz,
    int32_t* __restrict__ d_label,
    int*     __restrict__ d_count,
    int max_pts)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;

  if (d_det_mask && d_det_mask[idx] == 0) return;

  const int32_t obj_id = d_mask[idx];
  const float   depth  = d_depth[idx];
  const float   thresh = (obj_id == 0) ? max_bg_depth : max_obj_depth;

  if (depth <= 0.f || depth > thresh || !isfinite(depth)) return;

  const int slot = atomicAdd(d_count, 1);
  if (slot >= max_pts) return;

  const int row = idx / cols;
  const int col = idx % cols;

  d_xyz[3 * slot    ] = (col - cx) * depth * fx_inv;
  d_xyz[3 * slot + 1] = (row - cy) * depth * fy_inv;
  d_xyz[3 * slot + 2] = depth;
  d_label[slot]       = obj_id;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static void checkCuda(cudaError_t err, const char* where)
{
  if (err != cudaSuccess)
    fprintf(stderr, "[cuda_backproject] CUDA error at %s: %s\n",
            where, cudaGetErrorString(err));
}

// ── Public API ───────────────────────────────────────────────────────────────

void gpuBackprojectAlloc(GpuBackprojectScratch& s, int rows, int cols)
{
  if (s.d_depth_f) gpuBackprojectFree(s);

  s.rows    = rows;
  s.cols    = cols;
  s.max_pts = rows * cols;

  const int N = rows * cols;
  checkCuda(cudaMalloc(&s.d_depth_f,  N * sizeof(float)),   "alloc d_depth_f");
  checkCuda(cudaMalloc(&s.d_mask,     N * sizeof(int32_t)), "alloc d_mask");
  checkCuda(cudaMalloc(&s.d_det_mask, N * sizeof(uint8_t)), "alloc d_det_mask");
  checkCuda(cudaMalloc(&s.d_xyz,   3 * N * sizeof(float)),  "alloc d_xyz");
  checkCuda(cudaMalloc(&s.d_label,    N * sizeof(int32_t)), "alloc d_label");
  checkCuda(cudaMalloc(&s.d_count,        sizeof(int)),     "alloc d_count");
}

void gpuBackprojectFree(GpuBackprojectScratch& s)
{
  if (s.d_depth_f)  { cudaFree(s.d_depth_f);  s.d_depth_f  = nullptr; }
  if (s.d_mask)     { cudaFree(s.d_mask);      s.d_mask     = nullptr; }
  if (s.d_det_mask) { cudaFree(s.d_det_mask);  s.d_det_mask = nullptr; }
  if (s.d_xyz)      { cudaFree(s.d_xyz);       s.d_xyz      = nullptr; }
  if (s.d_label)    { cudaFree(s.d_label);     s.d_label    = nullptr; }
  if (s.d_count)    { cudaFree(s.d_count);     s.d_count    = nullptr; }
  s.rows = s.cols = s.max_pts = 0;
}

int gpuBackproject(
    GpuBackprojectScratch& s,
    const double*  h_depth,
    const int32_t* h_mask,
    const uint8_t* h_det_mask,
    int rows, int cols,
    float fx_inv, float fy_inv,
    float cx, float cy,
    float max_bg_depth, float max_obj_depth,
    float* h_xyz_out, int32_t* h_label_out)
{
  if (!s.d_depth_f || s.rows != rows || s.cols != cols) return 0;

  const int N = rows * cols;

  // DynoSAM stores Depth as double; convert to float before GPU upload.
  // thread_local avoids per-call heap allocation.
  static thread_local std::vector<float> depth_f;
  depth_f.resize(N);
  for (int i = 0; i < N; ++i) depth_f[i] = static_cast<float>(h_depth[i]);

  checkCuda(cudaMemcpy(s.d_depth_f, depth_f.data(), N * sizeof(float),
                       cudaMemcpyHostToDevice), "depth H2D");
  checkCuda(cudaMemcpy(s.d_mask, h_mask, N * sizeof(int32_t),
                       cudaMemcpyHostToDevice), "mask H2D");
  if (h_det_mask)
    checkCuda(cudaMemcpy(s.d_det_mask, h_det_mask, N * sizeof(uint8_t),
                         cudaMemcpyHostToDevice), "det_mask H2D");

  checkCuda(cudaMemset(s.d_count, 0, sizeof(int)), "reset count");

  const int threads = 256;
  const int blocks  = (N + threads - 1) / threads;
  backprojectKernel<<<blocks, threads>>>(
      s.d_depth_f, s.d_mask,
      h_det_mask ? s.d_det_mask : nullptr,
      rows, cols,
      fx_inv, fy_inv, cx, cy,
      max_bg_depth, max_obj_depth,
      s.d_xyz, s.d_label, s.d_count, s.max_pts);

  // cudaMemcpy D2H implicitly synchronises — no explicit cudaDeviceSynchronize needed.
  int n = 0;
  checkCuda(cudaMemcpy(&n, s.d_count, sizeof(int),
                       cudaMemcpyDeviceToHost), "count D2H");
  if (n > s.max_pts) n = s.max_pts;

  if (n > 0) {
    checkCuda(cudaMemcpy(h_xyz_out,   s.d_xyz,   3 * n * sizeof(float),
                         cudaMemcpyDeviceToHost), "xyz D2H");
    checkCuda(cudaMemcpy(h_label_out, s.d_label,     n * sizeof(int32_t),
                         cudaMemcpyDeviceToHost), "label D2H");
  }
  return n;
}

}  // namespace dyno::cuda
