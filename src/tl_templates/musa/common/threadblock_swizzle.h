#pragma once

namespace tl {

__device__ inline unsigned int ceil_div(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

template <int panel_width>
__device__ inline dim3 rasterization2DRow() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.x;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.x;
  const unsigned int col_idx = (panel_idx & 1)
                                   ? gridDim.x - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int row_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

template <int panel_width>
__device__ inline dim3 rasterization2DColumn() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.y;
  const unsigned int row_idx = (panel_idx & 1)
                                   ? gridDim.y - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

template <int swizzle_width>
__device__ inline dim3 swizzle2D() {
  const unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned int num_blocks_per_group = swizzle_width * gridDim.y;
  const unsigned int group_id = block_id / num_blocks_per_group;
  const unsigned int group_offset = block_id % num_blocks_per_group;
  const unsigned int bx = group_id * swizzle_width + (group_offset % swizzle_width);
  const unsigned int by = group_offset / swizzle_width;
  return {bx, by, blockIdx.z};
}

}  // namespace tl
