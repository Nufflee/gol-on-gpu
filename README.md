# Conway's Game of Life on the GPU

This is a WIP CUDA implementation of the [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) created to explore the wonderful world of [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) with a visualizer written using [SDL2](https://www.libsdl.org/).

## Current results
### Empty grid
- CPU: ~28 million cells/sec
- GPU (CUDA): ~3000 million cells/sec

Performance in this case is limited by the CPU <-> GPU memory transfer bandwidth which accounts for **~93%** of the runtime on an RTX 2080Ti. Measured banwidth on a PCIe 3.0 x8 slot is ~6.5 GB/s (~80% utilization), and ~14 GB/s (~88% utilization) on a PCIe 3.0 x16 slot.

### Randomized grid (50% live cells)
TODO...

## Building
TODO...