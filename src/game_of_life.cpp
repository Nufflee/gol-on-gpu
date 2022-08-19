#include "game_of_life.hpp"
#include <cassert>

Cell Board::get_cell(uint32_t x, uint32_t y) const {
	assert(x < m_Width && y < m_Height);

	return m_Cells[y * m_Width + x];
}

Cell Board::get_cell_or_dead(uint32_t x, uint32_t y) const {
	if (x >= m_Width || y >= m_Height) {
		return Cell::DEAD;
	}

	return m_Cells[y * m_Width + x];
}

void Board::set_cell(uint32_t x, uint32_t y, const Cell cell) {
	assert(x < m_Width && y < m_Height);

	m_Cells[y * m_Width + x] = cell;
}

void Board::clear() {
	for (uint32_t i = 0; i < m_Width * m_Height; i++) {
		m_Cells[i] = Cell::DEAD;
	}
}

Board& CPU_GameOfLife::step() {
	for (int y = 0; y < m_Height; y++) {
		for (int x = 0; x < m_Width; x++) {
			// MSVC is too dumb to unroll the loop properly, so we have to do it manually.
			int neighbors = 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x - 1, y - 1) == Cell::ALIVE ? 1 : 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x, y - 1) == Cell::ALIVE ? 1 : 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x + 1, y - 1) == Cell::ALIVE ? 1 : 0;

			neighbors += m_CurrentBoard.get_cell_or_dead(x - 1, y) == Cell::ALIVE ? 1 : 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x + 1, y) == Cell::ALIVE ? 1 : 0;

			neighbors += m_CurrentBoard.get_cell_or_dead(x - 1, y + 1) == Cell::ALIVE ? 1 : 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x, y + 1) == Cell::ALIVE ? 1 : 0;
			neighbors += m_CurrentBoard.get_cell_or_dead(x + 1, y + 1) == Cell::ALIVE ? 1 : 0;

			Cell current = m_CurrentBoard.get_cell(x, y);

			if (current == Cell::ALIVE) {
				if (neighbors == 2 || neighbors == 3) {
					m_NextBoard.set_cell(x, y, Cell::ALIVE);
				} else {
					m_NextBoard.set_cell(x, y, Cell::DEAD);
				}
			} else {
				if (neighbors == 3) {
					m_NextBoard.set_cell(x, y, Cell::ALIVE);
				} else {
					m_NextBoard.set_cell(x, y, Cell::DEAD);
				}
			}
		}
	}

	std::swap(m_CurrentBoard, m_NextBoard);

	return m_CurrentBoard;
}