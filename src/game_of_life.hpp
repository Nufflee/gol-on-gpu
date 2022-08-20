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
		: m_Width(width), m_Height(height) {
	}

	// Randomizes the board with a given probability of a cell being alive.
	void randomize(float prob = 0.5);

	virtual void set_cell(uint32_t x, uint32_t y, const Cell cell) = 0;
	void clear();

	// @returns The cell at the given position (x, y). Asserts if the position is out of the bounds of the board.
	virtual Cell get_cell(uint32_t x, uint32_t y) const = 0;
	// @returns The cell at the given position (x, y), or a dead cell if the position is outside the board.
	virtual Cell get_cell_or_dead(uint32_t x, uint32_t y) const = 0;

protected:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;
};

class GameOfLife {
public:
	GameOfLife(uint32_t width, uint32_t height)
		: m_Width(width), m_Height(height) {
	}

	virtual Board& step() = 0;
protected:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;
};

class CPU_Board : public Board {
public:
	CPU_Board(uint32_t width, uint32_t height)
		: Board(width, height), m_Cells(width * height) {
	}

	void set_cell(uint32_t x, uint32_t y, const Cell cell) override;

	Cell get_cell(uint32_t x, uint32_t y) const override;
	Cell get_cell_or_dead(uint32_t x, uint32_t y) const override;

private:
	std::vector<Cell> m_Cells;
};

class CPU_GameOfLife : public GameOfLife {
public:
	CPU_GameOfLife(uint32_t width, uint32_t height)
		: GameOfLife(width, height), m_CurrentBoard(width, height), m_NextBoard(width, height) {
	}

	CPU_Board& step();

	uint32_t width() const { return m_Width; }
	uint32_t height() const { return m_Height; }
	CPU_Board& current_board() { return m_CurrentBoard; }

private:
	CPU_Board m_CurrentBoard;
	CPU_Board m_NextBoard;
};