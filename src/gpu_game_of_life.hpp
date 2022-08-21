#pragma once

#include <string>
#include "game_of_life.hpp"
#include <cuda_runtime.h>

class GPU_Board : public Board {
public:
	GPU_Board(uint32_t width, uint32_t height);
	GPU_Board(GPU_Board&&);

	~GPU_Board();

	GPU_Board& operator=(GPU_Board&&);

	// NOTE: Only sets the cell in the CPU buffer.
	void set_cell(uint32_t x, uint32_t y, const Cell cell) override;

	// @returns The cell at the given position (x, y). Asserts if the position is out of the bounds of the board.
	Cell get_cell(uint32_t x, uint32_t y) const override;
	// @returns The cell at the given position (x, y), or a dead cell if the position is outside the board.
	Cell get_cell_or_dead(uint32_t x, uint32_t y) const override;

	void copy_host_to_device();
	void copy_device_to_host();

	inline uint32_t width() const { return m_Width; }
	inline uint32_t height() const { return m_Height; }
	inline Cell* host_cells() const { return m_HostCells; }
	struct cudaPitchedPtr device_cells() { return m_DeviceCells; }

private:
	Cell* m_HostCells = nullptr;
	cudaPitchedPtr m_DeviceCells {};
};

class GPU_GameOfLife {
public:
	GPU_GameOfLife(uint32_t width, uint32_t height)
		: m_CurrentBoard(width, height), m_NextBoard(width, height) {
	}

	GPU_Board& step();

	GPU_Board& current_board() { return m_CurrentBoard; }

private:
	GPU_Board m_CurrentBoard;
	GPU_Board m_NextBoard;
};

std::string get_device_name();