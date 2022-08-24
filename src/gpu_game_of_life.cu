#include "gpu_game_of_life.hpp"
#include <cassert>
#include <immintrin.h>
#include "time.hpp"

#define CHECK_CUDA_CALL(ret) check_cuda_call_impl(ret, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_CALL() check_cuda_call_impl(cudaGetLastError(), __FILE__, __LINE__)

void check_cuda_call_impl(const cudaError_t err, const char* fileName, const int lineNumber) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\n", fileName, lineNumber, cudaGetErrorString(err));
		exit(1);
	}
}

GPU_Board::GPU_Board(uint32_t width, uint32_t height)
	: Board(width, height) {
	CHECK_CUDA_CALL(cudaMallocHost(&m_HostCells, width * height * sizeof(uint8_t)));
	// Using pitched allocations supposedly makes memory accesses more efficient and coalesced, by aligning the allocated memory in such a way that the least number of memory transactions occurs while accessing it.
	CHECK_CUDA_CALL(cudaMalloc3D(&m_DeviceCells, cudaExtent { width * sizeof(uint8_t), height, 1 }));
}

GPU_Board::GPU_Board(GPU_Board&& other)
	: Board(0, 0) {
	*this = std::move(other);
}

GPU_Board::~GPU_Board() {
	CHECK_CUDA_CALL(cudaFreeHost(m_HostCells));
	CHECK_CUDA_CALL(cudaFree(m_DeviceCells.ptr));
}

GPU_Board& GPU_Board::operator=(GPU_Board&& other) {
	if (this != &other) {
		CHECK_CUDA_CALL(cudaFreeHost(m_HostCells));
		CHECK_CUDA_CALL(cudaFree(m_DeviceCells.ptr));

		m_Width = other.m_Width;
		m_Height = other.m_Height;
		m_HostCells = other.m_HostCells;
		m_DeviceCells = other.m_DeviceCells;

		other.m_HostCells = nullptr;
		other.m_DeviceCells = {};
	}

	return *this;
}

void GPU_Board::set_cell(uint32_t x, uint32_t y, const Cell cell) {
	assert(x < m_Width && y < m_Height);

	uint8_t* current = &m_HostCells[y * m_Width + x];

	*current = (*current & ~1) | cell;
}

Cell GPU_Board::get_cell(uint32_t x, uint32_t y) const {
	assert(x < m_Width && y < m_Height);

	return (Cell)(m_HostCells[y * m_Width + x] & 1);
}

Cell GPU_Board::get_cell_or_dead(uint32_t x, uint32_t y) const {
	if (x < m_Width && y < m_Height) {
		return (Cell)(m_HostCells[y * m_Width + x] & 1);
	}

	return Cell::DEAD;
}

void GPU_Board::copy_host_to_device() {
	CHECK_CUDA_CALL(cudaMemcpy2D(m_DeviceCells.ptr, m_DeviceCells.pitch, m_HostCells, m_Width * sizeof(uint8_t), m_Width * sizeof(uint8_t), m_Height, cudaMemcpyHostToDevice));
}

void GPU_Board::copy_device_to_host() {
	CHECK_CUDA_CALL(cudaMemcpy2D(m_HostCells, m_Width * sizeof(uint8_t), m_DeviceCells.ptr, m_DeviceCells.pitch, m_Width * sizeof(uint8_t), m_Height, cudaMemcpyDeviceToHost));
	m_CurrentGeneration = 0;
}

bool GPU_Board::shift_out_generation() {
	if (m_CurrentGeneration == 7 || m_CurrentGeneration == -1) {
		return false;
	}

	__m256i mask = _mm256_set1_epi8(255 >> 1);

	// TODO: Use cpuid intrinsic to check if AVX2 is supported, at runtime. https://docs.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=msvc-170
	for (uint32_t i = 0; i < m_Width * m_Height / 32; i++) {
		// NOTE: MSVC fails to autovectorize this loop so I had to vectorize it manually. Inspired by GCC's vectorization.
		__m256i simdData8 = _mm256_loadu_si256((__m256i*)&m_HostCells[i * 32]);

		// Right shifting a 16-bit value by 1 bit (which is also generalizable to any arbitrary shift n) and then masking
		// out the lower 7 bits of each individual 8-bit value is equivalent to right shifting the individual 8-bit values by 1 bit.
		// This identity is being taken advantage of here as AVX2 only supports 16-bit right shifts.
		__m256i result = _mm256_srli_epi16(simdData8, 1);
		result = _mm256_and_si256(result, mask);

		_mm256_storeu_si256((__m256i*)&m_HostCells[i * 32], result);
	}

	for(int i = 0; i < m_Width * m_Height % 32; i++) {
		m_HostCells[i] >>= 1;
	}

	m_CurrentGeneration += 1;

	return true;
}

void GPU_Board::reset_generation() {
	m_CurrentGeneration = -1;
}

__device__ Cell get_cell(const cudaPitchedPtr board, uint32_t x, uint32_t y, int generation) {
	return (Cell)(((uint8_t*)board.ptr)[y * board.pitch + x] >> generation & 1);
}

__device__ void set_cell(const cudaPitchedPtr board, uint32_t x, uint32_t y, const Cell cell, int generation) {
	uint8_t* current = &((uint8_t*)board.ptr)[y * board.pitch + x];

	*current = (*current & ~(1 << generation)) | (cell << generation);
}

#define BRANCHLESS true

// TODO: Try to coalesce memory accesses
__global__ void game_of_life_kernel(cudaPitchedPtr generations, uint32_t boardWidth, uint32_t boardHeight, int generation) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	int neighbors = 0;

	if (y > 0) {
		neighbors += get_cell(generations, x, y - 1, generation - 1);

		if (x > 0) {
			neighbors += get_cell(generations, x - 1, y - 1, generation - 1);
		}

		if (x < boardWidth - 1) {
			neighbors += get_cell(generations, x + 1, y - 1, generation - 1);
		}
	}

	if (x > 0) {
		neighbors += get_cell(generations, x - 1,  y, generation - 1);
	}

	if (x < boardWidth - 1) {
		neighbors += get_cell(generations, x + 1, y, generation - 1);
	}

	if (y < boardHeight - 1) {
		neighbors += get_cell(generations, x, y + 1, generation - 1);

		if (x > 0) {
			neighbors += get_cell(generations, x - 1, y + 1, generation - 1);
		}

		if (x < boardWidth - 1) {
			neighbors += get_cell(generations, x + 1, y + 1, generation - 1);
		}
	}

	Cell current = get_cell(generations, x, y, generation - 1);

#if BRANCHLESS == true
	set_cell(generations, x, y, (Cell)((int)current * (neighbors == 2 || neighbors == 3) + (1 - (int)current) * (neighbors == 3)), generation);
#else
	if (current == Cell::ALIVE) {
		if (neighbors == 2 || neighbors == 3) {
			set_cell(generations, x, y, Cell::ALIVE, generation);
		} else {
			set_cell(generations, x, y, Cell::DEAD, generation);
		}
	} else {
		if (neighbors == 3) {
			set_cell(generations, x, y, Cell::ALIVE, generation);
		} else {
			set_cell(generations, x, y, Cell::DEAD, generation);
		}
	}
#endif
}

#define BANDWIDTH_MEASUREMENT false

// TODO: Make use of all 7 new generations
// TODO: Run the kernel asynchronously and put generations in some sort of a queue
GPU_Board& GPU_GameOfLife::step() {
	if (m_CurrentBoard.shift_out_generation()) {
		return m_CurrentBoard;
	} else {
		// TODO: Check the maximum/best number of blocks for current GPU
		constexpr int BLOCK_SIZE = 32;
		dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

		uint32_t width = m_CurrentBoard.width();
		uint32_t height = m_CurrentBoard.height();

		dim3 blockCount((int)ceil((float)width / BLOCK_SIZE), (int)ceil((float)height / BLOCK_SIZE));

#if BANDWIDTH_MEASUREMENT == true
		auto start = get_time_secs();
#endif
		m_CurrentBoard.copy_host_to_device();
#if BANDWIDTH_MEASUREMENT == true
		auto end = get_time_secs();
		auto time = end - start;
		printf("H2D time: %f sec, bandwidth: %f GB/s\n", time, (width * height * sizeof(uint8_t) / 1e9) / time);
#endif

		for (int i = 1; i < 8; i++) {
			game_of_life_kernel<<<blockCount, blockDim>>>(m_CurrentBoard.device_cells(), width, height, i);
			CHECK_LAST_CUDA_CALL();
		}

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

#if BANDWIDTH_MEASUREMENT == true
		start = get_time_secs();
#endif
		m_CurrentBoard.copy_device_to_host();
#if BANDWIDTH_MEASUREMENT == true
		end = get_time_secs();
		time = end - start;
		printf("D2H time: %f sec, bandwidth: %f GB/s\n", time, (width * height * sizeof(uint8_t) / 1e9) / time);
#endif

		m_CurrentBoard.shift_out_generation();

		return m_CurrentBoard;
	}
}

std::string get_device_name() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	return std::string(prop.name);
}