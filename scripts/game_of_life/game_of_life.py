#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import os
import pickle
from dataclasses import dataclass
from typing import ClassVar, List, Optional
from threading import Thread
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from tqdm import tqdm
from typeguard import typechecked
import time
import pandas as pd
import subprocess
import streamlit
from libs.benchmark import benchmark
from libs.logger import configure_logging

@dataclass
class GameOfLife:
    """
    The Conway's Game of Life: This class implements the game on a cellular automaton.
    It uses a numpy array to represent the state of the game.
    """

    # the state (matrix)
    state: np.ndarray

    # the current generation
    generation: int = 0

    # the max number of generations
    max_generations: int = 100

    # the representation of a dead cell
    dead_cell_char: str = " "

    # the representatoin of a live cell
    live_cell_char: str = "࿕"

    # class constant
    ALIVE: ClassVar[int] = 1
    DEAD: ClassVar[int] = 0

    # kernel to count the neighbors
    NEIGHBOR_KERNEL = np.array(
        [
            [1, 1, 1],  #
            [1, 0, 1],  #
            [1, 1, 1],  #
        ],
        dtype=np.uint8,
    )

    GLIDER_PATTERN = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    BLINKER_PATTERN = np.array([
        [1, 1, 1]
    ], dtype=np.uint8)

    @typechecked
    def __post_init__(self) -> None:
        """Post-initialization checks for the Game of Life class."""

        # the state need to be a numpy array
        if not isinstance(self.state, np.ndarray):
            raise TypeError("Game of Life must be a numpy array")

        # convert the state to a numpy array of uint8
        if self.state.dtype != np.uint8:
            self.state = self.state.astype(np.uint8)

        # the state need to be a 2D numpy array
        if self.state.ndim != 2:
            raise TypeError("Game of Life must be a 2D array")

        # the state need to be a 0 (DEAD) or 1 (ALIVE)
        if not np.all(np.isin(self.state, [self.ALIVE, self.DEAD])):
            raise TypeError("Game of Life must contain only cells 0 and 1")

        # the max_generations need to be positive
        if self.max_generations <= 0:
            raise TypeError("max_iterations must be greater than 0")

    @classmethod
    @typechecked
    def from_list(cls, initial_state: List[List[int]]) -> "GameOfLife":
        """Create a GameOfLife object from a list of lists."""

        # validate the input
        if not initial_state:
            raise ValueError("Initial state cannot be empty")

        # the input should be a list of lists
        if not all(isinstance(row, list) for row in initial_state):
            raise ValueError("Initial state must be a list of lists")

        return cls(state=np.array(initial_state, dtype=np.uint8))

    # @typechecked
    def population(self) -> np.uint64:
        """sum all the ALIVE cells."""
        return np.sum(self.state, dtype=np.uint64)

    @typechecked
    def __str__(self) -> str:
        """Return a string representation of the current state."""
        grid = "\n".join(
            "".join(
                self.live_cell_char if cell == self.ALIVE else self.dead_cell_char
                for cell in row
            )
            for row in self.state
        )
        header = (
            f"Generation: {self.generation:04d} | Population: {self.population():04d}\n"
        )
        separator = "-" * max(len(line) for line in grid.split("\n")) + "\n"
        return header + separator + grid

    # @typechecked
    # noinspection PyMethodMayBeStatic
    def _compress(self, state: np.ndarray) -> np.ndarray:
        """Compress the state by removing border rows and columns that contain only dead cells (zeros)."""

        if state.size == 0:
            return np.zeros((1, 1), dtype=state.dtype)

        # find rows and columns with any live cells
        live_rows = np.any(state != 0, axis=1)
        live_cols = np.any(state != 0, axis=0)

        if not live_rows.any() or not live_cols.any():
            return np.zeros((1, 1), dtype=state.dtype)

        # get bounding box indices
        row_min, row_max = np.where(live_rows)[0][[0, -1]]
        col_min, col_max = np.where(live_cols)[0][[0, -1]]

        return state[row_min : row_max + 1, col_min : col_max + 1]

    # @typechecked
    def evolve(self) -> "GameOfLife":
        """Evolve the current state to the next generation."""

        # padded state around the original state
        padded_state = np.pad(
            self.state,
            pad_width=1,
            mode="constant",
            constant_values=self.DEAD,  # fill with dead cells
        )

        # count the neighbors using convolution
        neighbors = convolve(
            padded_state,
            self.NEIGHBOR_KERNEL,
            mode="constant",
            cval=self.DEAD,  # fill with dead cells
        )

        # determine which cells are alive
        alive = padded_state == self.ALIVE

        # apply the rules of the Game of Life vectorized
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html
        new_state = np.where(
            (alive & ((neighbors == 2) | (neighbors == 3)))
            | (~alive & (neighbors == 3)),
            self.ALIVE,
            self.DEAD,
        ).astype(np.uint8)

        # assign the new_state pos-compression
        self.state = self._compress(new_state)

        # increment the generation
        self.generation += 1
        return self

    def detect_patterns(self) -> None:
        """
        Detect gliders and blinkers in the current state.
        Logs detection if found.
        """
        log = logging.getLogger(__name__)
        state = self.state

        # detect Gliders
        glider_variations = [
            self.GLIDER_PATTERN,
            np.rot90(self.GLIDER_PATTERN),
            np.rot90(self.GLIDER_PATTERN, 2),
            np.rot90(self.GLIDER_PATTERN, 3),
            np.fliplr(self.GLIDER_PATTERN),
            np.flipud(self.GLIDER_PATTERN),
            np.fliplr(np.rot90(self.GLIDER_PATTERN)),
            np.flipud(np.rot90(self.GLIDER_PATTERN)),
        ]

        rows, cols = state.shape
        for i in range(rows - 2):
            for j in range(cols - 2):
                submatrix = state[i:i+3, j:j+3]
                for var in glider_variations:
                    if np.array_equal(submatrix, var):
                        log.info(f"Glider detected at position ({i},{j}), Generation: {self.generation}")

        # Detect Blinkers
        blinker_variations = [
            self.BLINKER_PATTERN,
            self.BLINKER_PATTERN.T  # vertical version
        ]

        # Check horizontal blinkers (1x3)
        for i in range(rows):
            for j in range(cols - 2):
                submatrix = state[i:i + 1, j:j + 3]
                if np.array_equal(submatrix, self.BLINKER_PATTERN):
                    log.info(f"Blinker (H) detected at ({i},{j}), Generation: {self.generation}")

        # Check vertical blinkers (3x1)
        for i in range(rows - 2):
            for j in range(cols):
                submatrix = state[i:i + 3, j:j + 1]
                if np.array_equal(submatrix, self.BLINKER_PATTERN.T):
                    log.info(f"Blinker (V) detected at ({i},{j}), Generation: {self.generation}")
    # @typechecked
    def evolve_parallel(self, num_threads: int) -> "GameOfLife":
        """
        Evolve the current state to the next generation using multi-threading.
        """

        # pad the state around the edges with DEAD cells
        padded_state = np.pad(
            self.state,
            pad_width=1,
            mode="constant",
            constant_values=self.DEAD,
        )

        height, width = self.state.shape

        # define the size of each chunk (rows per thread)
        chunk_height = height // num_threads

        # storage for each thread's result
        results = [None] * num_threads

        def worker(start_row: int, end_row: int, idx: int):
            """
            Worker function to process a block of rows.
            Handles overlapping edges correctly.
            """
            # Para chunks intermedios, agrega una fila arriba y una abajo
            # Para el primer bloque no hay fila arriba real -> ya está paddeado
            # Para el último bloque no hay fila abajo real -> ya está paddeado
            block_start = max(start_row, 0)
            block_end = min(end_row + 2, padded_state.shape[0])  # +2 por padding

            block = padded_state[block_start:block_end, :]

            # Count neighbors
            neighbors = convolve(
                block,
                self.NEIGHBOR_KERNEL,
                mode="constant",
                cval=self.DEAD,
            )

            alive = block == self.ALIVE

            new_block = np.where(
                (alive & ((neighbors == 2) | (neighbors == 3)))
                | (~alive & (neighbors == 3)),
                self.ALIVE,
                self.DEAD,
            ).astype(np.uint8)

            # Remove overlap
            if idx == 0:
                results[idx] = new_block[:-1, :]  # solo quitar fila inferior
            elif idx == num_threads - 1:
                results[idx] = new_block[1:, :]  # solo quitar fila superior
            else:
                results[idx] = new_block[1:-1, :]  # quitar fila superior e inferior

        # create and start all threads
        threads = []
        for i in range(num_threads):
            start_row = i * chunk_height
            # last chunk takes remaining rows
            end_row = (i + 1) * chunk_height if i < num_threads - 1 else height
            t = Thread(target=worker, args=(start_row, end_row, i))
            threads.append(t)
            t.start()

        # wait for all threads to finish
        for t in threads:
            t.join()

        # combine results from all threads
        new_state = np.vstack(results)

        # compress the new state to remove empty borders
        self.state = self._compress(new_state)

        # increment the generation counter
        self.generation += 1

        # detect a glider
        self.detect_patterns()

        return self



    @typechecked
    def run_simulation(
        self,
        max_generations: Optional[int] = None,
        show_progress: Optional[bool] = False,
    ) -> str:
        """Run the simulation for a given number of generations."""

        # get the logger
        log = logging.getLogger(__name__)

        if max_generations is None:
            max_generations = self.max_generations

        if max_generations <= 0:
            raise ValueError("max_generations must be positive")

        # list to save the statistics of each iteration
        stats = []

        # iterate over the number of generations
        for i in tqdm(
            range(0, max_generations),
            desc="Evolving generations",
            unit="gen",
            ncols=200,  # progress bar width
            disable=not show_progress,
        ):
            # save the previous state
            previous_state = self.state.copy()

            # generate the next generation and took its execution time
            #self.evolve()
            start_time = time.perf_counter()
            self.evolve_parallel(num_threads=6)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # I take the statistics of each iteration
            stat = {
                "iteration": i + 1,
                "live_cells": int(self.population()),
                "dead_cells": int(self.state.size - self.population()),
                "duration_ms": duration_ms,
            }

            # add the statistics of each iteration to the list
            stats.append(stat)

            # detect stable states (no changes) and stop the simulation.
            if np.array_equal(previous_state, self.state):
                print(f"INFO: Stable state detected at generation {self.generation}. Stopping simulation")
                break

            # if the population is 0, break the loop
            if self.population() == 0:
                print(f"INFO: No live cells detected at generation {self.generation}.")
                return "WARN: Stopping simulation at:\n" + str(self)

        # the data dir
        data_dir = os.path.join("..", "..", "data")

        # convert to DataFrame and save to CSV
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(os.path.join(data_dir, "stats.csv"), index=False)
        # path load_stats.py
        load_stats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql", "load_stats.py"))

        # I enter the simulation data into the database using a subprocess
        subprocess.run(["python", load_stats_path], check=True)

        return str(self)


# @typechecked
# def plot_game_of_life(game_of_life: GameOfLife, path: Optional[str] = None) -> None:
#     """Draw the Game of Life using matplotlib and pyplot."""
#
#     # define the figure
#     fig = plt.figure(facecolor="white", dpi=200)
#
#     # get the current axis
#     ax = plt.gca()
#
#     # hide the axis
#     ax.set_axis_off()
#
#     sns.heatmap(
#         game_of_life.state,  # ndarray
#         cmap="binary",
#         cbar=False,
#         square=True,
#         linewidths=0.25,
#         linecolor="#f0f0f0",  # rgb
#         ax=ax,
#     )
#
#     # set the title
#     plt.title("The Conway's Game of Life")
#
#     # create some stats
#     # the total of space inside the grid
#     total_space = game_of_life.state.shape[0] * game_of_life.state.shape[1]
#     density = game_of_life.population() / total_space
#
#     stats = (
#         f"Generation: {game_of_life.generation}\n"
#         f"Population: {game_of_life.population()}\n"
#         f"Grid size: {game_of_life.state.shape[0]} x {game_of_life.state.shape[1]}\n"
#         f"Density: {density:.2f}"
#     )
#
#     # plot the stats
#     plt.figtext(
#         0.99,
#         0.01,
#         stats,
#         horizontalalignment="right",
#         verticalalignment="bottom",
#         fontsize=10,
#         bbox=dict(facecolor="white", alpha=0.8, boxstyle="round", pad=0.5),
#     )
#
#     # compress the layout
#     plt.tight_layout()
#
#     # we need to save the plot?
#     if path is not None:
#         plt.savefig(
#             f"{path}/game_of_life-{game_of_life.generation:04d}.png",
#             dpi=200,
#             bbox_inches="tight",
#         )
#
#     # show time !
#     plt.show()
#


def plot_game_of_life(game_of_life, path: Optional[str] = None):
        """Draw the Game of Life using matplotlib and pyplot."""

        # define the figure and axis
        fig, ax = plt.subplots(facecolor="white", dpi=200)

        # hide the axis
        ax.set_axis_off()

        sns.heatmap(
            game_of_life.state,
            cmap="binary",
            cbar=False,
            square=True,
            linewidths=0.25,
            linecolor="#f0f0f0",
            ax=ax,
        )

        # set the title
        ax.set_title("The Conway's Game of Life")

        # create some stats
        total_space = game_of_life.state.shape[0] * game_of_life.state.shape[1]
        density = game_of_life.population() / total_space

        stats = (
            f"Generation: {game_of_life.generation}\n"
            f"Population: {game_of_life.population()}\n"
            f"Grid size: {game_of_life.state.shape[0]} x {game_of_life.state.shape[1]}\n"
            f"Density: {density:.2f}"
        )

        # plot the stats
        fig.text(
            0.99,
            0.01,
            stats,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round", pad=0.5),
        )

        # compress the layout
        fig.tight_layout()

        # save the plot if needed
        if path is not None:
            fig.savefig(
                f"{path}/game_of_life-{game_of_life.generation:04d}.png",
                dpi=200,
                bbox_inches="tight",
            )

        return fig

def main():
    # configure the logger
    configure_logging(logging.DEBUG)

    # get the logger
    log = logging.getLogger(__name__)
    log.debug("Starting main ..")

    # initial state
    # state = [
    #     [1, 1, 0],
    #     [0, 1, 1],
    #     [0, 1, 0],
    # ]
    # initial state
    pickle_file = os.path.join("..","..","data","initial.pkl")
    with open(pickle_file, "rb") as f:
        state = pickle.load(f).tolist()
    # create the object GameOfLife
    game_of_life = GameOfLife.from_list(state)

    # print the initial state
    #print(game_of_life)

    # run the simulation
    max_generations = 3000
    log.debug(f"Running simulation for {max_generations} generations ...")
    with benchmark(operation_name="run_simulation", log=log):
        game_of_life.run_simulation(max_generations, show_progress=True)

    # print the final state
    # print(game_of_life)
    # plot_game_of_life(game_of_life)
    # plot_game_of_life(game_of_life, "../../output/")

    log.debug("Done.")


if __name__ == "__main__":
    # Run the main in a profile fashion
    # cProfile.run("main()", "../../output/game_of_life.prof")
    main()