#include "gpu_game_of_life.hpp"
#include <cassert>
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

	m_HostCells[y * m_Width + x] = cell;
}

Cell GPU_Board::get_cell(uint32_t x, uint32_t y) const {
	assert(x < m_Width && y < m_Height);

	return m_HostCells[y * m_Width + x];
}

Cell GPU_Board::get_cell_or_dead(uint32_t x, uint32_t y) const {
	if (x < m_Width && y < m_Height) {
		return m_HostCells[y * m_Width + x];
	}

	return Cell::DEAD;
}

void GPU_Board::copy_host_to_device() {
	CHECK_CUDA_CALL(cudaMemcpy2D(m_DeviceCells.ptr, m_DeviceCells.pitch, m_HostCells, m_Width * sizeof(uint8_t), m_Width * sizeof(uint8_t), m_Height, cudaMemcpyHostToDevice));
}

void GPU_Board::copy_device_to_host() {
	CHECK_CUDA_CALL(cudaMemcpy2D(m_HostCells, m_Width * sizeof(uint8_t), m_DeviceCells.ptr, m_DeviceCells.pitch, m_Width * sizeof(uint8_t), m_Height, cudaMemcpyDeviceToHost));
}

#define BRANCHLESS true

__device__ Cell get_cell(const cudaPitchedPtr board, uint32_t x, uint32_t y) {
	return (Cell)(((uint8_t*)board.ptr)[y * board.pitch + x]);
}

__device__ void set_cell(const cudaPitchedPtr board, uint32_t x, uint32_t y, const Cell cell) {
	((uint8_t*)board.ptr)[y * board.pitch + x] = cell;
}

// TODO: Consolidate input and output boards into one array
__global__ void game_of_life_kernel(cudaPitchedPtr input, cudaPitchedPtr output, uint32_t boardWidth, uint32_t boardHeight) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	int neighbors = 0;

	if (y > 0) {
		neighbors += get_cell(input, x, y - 1);

		if (x > 0) {
			neighbors += get_cell(input, x - 1, y - 1);
		}

		if (x < boardWidth - 1) {
			neighbors += get_cell(input, x + 1, y - 1);
		}
	}

	if (x > 0) {
		neighbors += get_cell(input, x - 1,  y);
	}

	if (x < boardWidth - 1) {
		neighbors += get_cell(input, x + 1, y);
	}

	if (y < boardHeight - 1) {
		neighbors += get_cell(input, x, y + 1);

		if (x > 0) {
			neighbors += get_cell(input, x - 1, y + 1);
		}

		if (x < boardWidth - 1) {
			neighbors += get_cell(input, x + 1, y + 1);
		}
	}

	Cell current = get_cell(input, x, y);

#if BRANCHLESS == true
	set_cell(output, x, y, (int)current * (neighbors == 2 || neighbors == 3) + (1 - (int)current) * (neighbors == 3));
#else
	if (current == Cell::ALIVE) {
		if (neighbors == 2 || neighbors == 3) {
			set_cell(output, x, y, Cell::ALIVE);
		} else {
			set_cell(output, x, y, Cell::DEAD);
		}
	} else {
		if (neighbors == 3) {
			set_cell(output, x, y, Cell::ALIVE);
		} else {
			set_cell(output, x, y, Cell::DEAD);
		}
	}
#endif
}

#define BANDWIDTH_MEASUREMENT false

GPU_Board& GPU_GameOfLife::step() {
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

	game_of_life_kernel<<<blockCount, blockDim>>>(m_CurrentBoard.device_cells(), m_NextBoard.device_cells(), width, height);
	CHECK_LAST_CUDA_CALL();

	CHECK_CUDA_CALL(cudaDeviceSynchronize());

#if BANDWIDTH_MEASUREMENT == true
	start = get_time_secs();
#endif
	m_NextBoard.copy_device_to_host();
#if BANDWIDTH_MEASUREMENT == true
	end = get_time_secs();
	time = end - start;
	printf("D2H time: %f sec, bandwidth: %f GB/s\n", time, (width * height * sizeof(uint8_t) / 1e9) / time);
#endif

	std::swap(m_CurrentBoard, m_NextBoard);

	return m_CurrentBoard;
}

std::string get_device_name() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	return std::string(prop.name);
}