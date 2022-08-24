#include "game_of_life.hpp"
#include "gpu_game_of_life.hpp"
#include "time.hpp"
#include <SDL2/SDL.h>

void benchmark() {
	{
		constexpr int size = 1 << 13;
		constexpr int n = 10;

		CPU_GameOfLife gol(size, size);

		gol.current_board().randomize();

		auto start = get_time_secs();
		for (int i = 0; i < n; i++) {
			Board& b = gol.step();
		}
		auto end = get_time_secs();

		printf("CPU:\n");
		printf("\t%dx%d randomized grid\n", size, size);
		printf("\t%.2f MCells/sec\n", 1 / ((end - start) / n / (size * size)) / 1e6);
	}

	{
		constexpr int size = 1 << 13;
		constexpr int n = 100;

		GPU_GameOfLife gol(size, size);

		double totalTime = 0;

		gol.current_board().randomize();

		for (int i = 0; i < n; i++) {
			auto start = get_time_secs();

			GPU_Board& b = gol.step();

			auto end = get_time_secs();
			totalTime += end - start;

			b.reset_generation();
		}

		printf("GPU (CUDA):\n");
		printf("\tName: %s\n", get_device_name().c_str());
		printf("\t%dx%d randomized grid\n", size, size);
		printf("\t%.2f MCells/sec\n", 1 / (totalTime / n / (size * size)) / 1e6 * 8);
		printf("\t%.2f msec/step\n", totalTime / n * 1000.0);
	}
}

// TODO: Add command line setting for pixels/cell.
int main(int argc, char* argv[]) {
	benchmark();

	SDL_Init(SDL_INIT_VIDEO);

	constexpr int WIDTH = 1920;
	constexpr int HEIGHT = 1080;
	constexpr int CELL_SIZE = 5;

	SDL_Window* window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);

	if (window == nullptr) {
		printf("Failed to create window: %s\n", SDL_GetError());
		return 1;
	}

	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	GPU_GameOfLife gol(WIDTH / CELL_SIZE, HEIGHT / CELL_SIZE);

	// Blinker
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2, HEIGHT / CELL_SIZE / 2 + 1, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2, HEIGHT / CELL_SIZE / 2, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2, HEIGHT / CELL_SIZE / 2 - 1, Cell::ALIVE);

	// Toad
	int offsetX = 10;
	int offsetY = 10;

	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 3 + offsetX, HEIGHT / CELL_SIZE / 2 - 1 + offsetY, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 2 + offsetX, HEIGHT / CELL_SIZE / 2 - 1 + offsetY, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 1 + offsetX, HEIGHT / CELL_SIZE / 2 - 1 + offsetY, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 2 + offsetX, HEIGHT / CELL_SIZE / 2 + offsetY, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 1 + offsetX, HEIGHT / CELL_SIZE / 2 + offsetY, Cell::ALIVE);
	gol.current_board().set_cell(WIDTH / CELL_SIZE / 2 - 0 + offsetX, HEIGHT / CELL_SIZE / 2 + offsetY, Cell::ALIVE);

	// Glider
	gol.current_board().set_cell(0, 1, Cell::ALIVE);
	gol.current_board().set_cell(1, 2, Cell::ALIVE);
	gol.current_board().set_cell(2, 0, Cell::ALIVE);
	gol.current_board().set_cell(2, 1, Cell::ALIVE);
	gol.current_board().set_cell(2, 2, Cell::ALIVE);

	gol.current_board().randomize();

	printf("Width: %d, Height: %d\n", gol.current_board().width(), gol.current_board().height());

	bool running = true;

	while (running) {
		SDL_Event event;

		while (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT: {
					running = false;
					break;
				}
			}
		}

		SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
		SDL_RenderClear(renderer);

		// Grid
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		for (int x = 0; x < WIDTH; x += CELL_SIZE) {
			SDL_RenderDrawLine(renderer, x, 0, x, HEIGHT);
		}
		for (int y = 0; y < HEIGHT; y += CELL_SIZE) {
			SDL_RenderDrawLine(renderer, 0, y, WIDTH, y);
		}

		// Cells
		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		for (int x = 0; x < WIDTH; x += CELL_SIZE) {
			for (int y = 0; y < HEIGHT; y += CELL_SIZE) {
				if (gol.current_board().get_cell(x / CELL_SIZE, y / CELL_SIZE) == Cell::ALIVE) {
					SDL_Rect rect = {x + 1, y + 1, CELL_SIZE - 1, CELL_SIZE - 1};
					SDL_RenderFillRect(renderer, &rect);
				}
			}
		}

		gol.step();

		SDL_RenderPresent(renderer);
		SDL_Delay(100);
	}

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);

	SDL_Quit();

	return 0;
}