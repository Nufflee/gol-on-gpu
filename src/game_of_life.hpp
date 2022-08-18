#pragma once

#include <cstdint>
#include <vector>

enum Cell : uint8_t {
	DEAD = 0,
	ALIVE = 1
};

class Board {
public:
	Board(uint32_t width, uint32_t height)
		: m_Width(width), m_Height(height), m_Cells(width * height) {
	}

	void clear();

	// @returns The cell at the given position (x, y). Asserts if the position is out of the bounds of the board.
	Cell get_cell(uint32_t x, uint32_t y) const;
	// @returns The cell at the given position (x, y), or a dead cell if the position is outside the board.
	Cell get_cell_or_dead(uint32_t x, uint32_t y) const;
	void set_cell(uint32_t x, uint32_t y, const Cell cell);

private:
	std::vector<Cell> m_Cells;
	uint32_t m_Width;
	uint32_t m_Height;
};

class GameOfLife {
public:
	GameOfLife(uint32_t width, uint32_t height)
		: m_Width(width), m_Height(height) {
	}

	virtual Board& step() = 0;
protected:
	uint32_t m_Width;
	uint32_t m_Height;
};

class CPU_GameOfLife : public GameOfLife {
public:
	CPU_GameOfLife(uint32_t width, uint32_t height)
		: GameOfLife(width, height), m_CurrentBoard(width, height), m_NextBoard(width, height) {
	}

	Board& step();

	uint32_t width() const { return m_Width; }
	uint32_t height() const { return m_Height; }
	Board& current_board() { return m_CurrentBoard; }

private:
	Board m_CurrentBoard;
	Board m_NextBoard;
};