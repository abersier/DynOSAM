#pragma once
#include <cstdint>

// GPU-accelerated pinhole backprojection for dense labelled cloud generation.
// Mirrors the structure of dyno_mppi_critic/cuda_sdf_query.cuh.
// CPU reference implementation: Frame_cpu.cc::projectToDenseCloud.

namespace dyno::cuda {

// Pre-allocated device buffers — sized once on first use, reused every frame.
struct GpuBackprojectScratch {
  float*    d_depth_f{nullptr};   // float copy of double depth  (rows × cols)
  int32_t*  d_mask{nullptr};      // ObjectId motion mask         (rows × cols)
  uint8_t*  d_det_mask{nullptr};  // detection mask, optional     (rows × cols)
  float*    d_xyz{nullptr};       // compact output: x,y,z packed (3 × max_pts)
  int32_t*  d_label{nullptr};     // compact output: object IDs   (max_pts)
  int*      d_count{nullptr};     // atomic write counter          (1 element)
  int rows{0}, cols{0};
  int max_pts{0};
};

// Allocate all device buffers for an image of size rows × cols.
// Frees any previous allocation in s first.
void gpuBackprojectAlloc(GpuBackprojectScratch& s, int rows, int cols);
void gpuBackprojectFree(GpuBackprojectScratch& s);

// Upload inputs, run kernel, download compact results.
// h_det_mask may be nullptr (detection mask disabled).
// h_xyz_out   must be pre-allocated by caller: float[3 * s.max_pts]
// h_label_out must be pre-allocated by caller: int32_t[s.max_pts]
// Returns number of valid points written.
int gpuBackproject(
    GpuBackprojectScratch& s,
    const double*  h_depth,       // DynoSAM stores Depth = double
    const int32_t* h_mask,        // ObjectId = int
    const uint8_t* h_det_mask,
    int rows, int cols,
    float fx_inv, float fy_inv,
    float cx, float cy,
    float max_bg_depth,
    float max_obj_depth,
    float*   h_xyz_out,
    int32_t* h_label_out);

}  // namespace dyno::cuda
