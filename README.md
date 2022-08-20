# Conway's Game of Life on the GPU

This is a WIP CUDA implementation of the [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) created to explore the wonderful world of [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) with a visualizer written using [SDL2](https://www.libsdl.org/).

## Current results
All performance measured on an Intel i9 9900K and an RTX 2080Ti (PCIe 3.0 **x8**).

### Empty grid (8192x8192)
- CPU: ~60 million cells/sec
- GPU (CUDA): ~3000 million cells/sec

Performance in this case is limited by the CPU <-> GPU memory transfer bandwidth which accounts for **~93%** of the runtime. Measured banwidth on a PCIe 3.0 x8 slot is ~6.5 GB/s (~80% utilization), and ~14 GB/s (~88% utilization) on a PCIe 3.0 x16 slot which leads to a significant increase in performance, up to 4300 million cells/sec.

### Randomized grid (32768x32768, 50% live cells)
- CPU: ~60 million cells/sec
- GPU (CUDA): ~3000 million cells/sec

Performance is again limited by the CPU <-> GPU memory transfer bandwidth, and therefore largely independent of the grid size. The GPU performance being no less than the empty grid case despite unpredictable branches is likely a testament to `nvcc`'s kernel optimization ability.

## Building
TODO...