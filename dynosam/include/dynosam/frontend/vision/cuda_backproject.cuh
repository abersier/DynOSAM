#pragma once
#include <cstdint>

namespace dyno::cuda {

// Pre-allocated device buffers — call gpuBackprojectAlloc once per resolution,
// gpuBackprojectFree on teardown.
struct GpuBackprojectScratch {
  float*   d_depth_f  {nullptr};
  int32_t* d_mask     {nullptr};
  uint8_t* d_det_mask {nullptr};
  float*   d_xyz      {nullptr};
  int32_t* d_label    {nullptr};
  int*     d_count    {nullptr};
  int rows{0}, cols{0}, max_pts{0};
};

void gpuBackprojectAlloc(GpuBackprojectScratch& s, int rows, int cols);
void gpuBackprojectFree(GpuBackprojectScratch& s);

// Returns the number of valid points written into h_xyz_out / h_label_out.
// h_xyz_out must hold at least rows*cols*3 floats.
// h_label_out must hold at least rows*cols int32_ts.
// Pass h_det_mask = nullptr to skip the detection-mask gate.
// DynoSAM depth is stored as double; conversion to float is done inside.
int gpuBackproject(
    GpuBackprojectScratch& s,
    const double*  h_depth,
    const int32_t* h_mask,
    const uint8_t* h_det_mask,
    int rows, int cols,
    float fx_inv, float fy_inv,
    float cx, float cy,
    float max_bg_depth, float max_obj_depth,
    float* h_xyz_out, int32_t* h_label_out);

}  // namespace dyno::cuda
