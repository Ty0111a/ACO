import random
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist


class Graph:
    def __init__(self):
        self.cords = np.empty(shape=(0, 0), dtype="double")
        self.distance_matrix = np.empty(shape=(0, 0), dtype="double")
        self.closeness_matrix = np.empty(shape=(0, 0), dtype="double")
        self.pheromone_matrix = np.empty(shape=(0, 0), dtype="double")

    def load(self, path, ph=0):
        temp = []
        with open(path, 'r') as file:
            for line in file:
                temp.append([float(i) for i in line.split(" ")])

        self.cords = np.array(temp)
        self.distance_matrix = cdist(self.cords, self.cords, 'euclidean')
        self.pheromone_matrix = np.full((len(self.cords), len(self.cords)), ph, dtype="double")

    def add_k_nearest_edges(self, k):
        if k >= len(self.cords) - 1:
            k = len(self.cords) - 2
        for i in self.distance_matrix:
            temp = np.sort(i)[k + 1]
            i[i > temp] = 0

        # recover lost values
        n = len(self.distance_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                self.distance_matrix[j][i] = max(self.distance_matrix[i][j], self.distance_matrix[j][i])
                self.distance_matrix[i][j] = self.distance_matrix[j][i]

        self.closeness_matrix = self.distance_matrix.copy()
        for i in range(len(self.closeness_matrix)):
            for j in range(len(self.closeness_matrix)):
                if self.closeness_matrix[i][j] == 0:
                    self.closeness_matrix[i][j] = 0
                    continue
                self.closeness_matrix[i][j] = 200 / self.closeness_matrix[i][j]

    def __len__(self):
        return len(self.cords)

    def evaporation(self, evaporation):
        self.pheromone_matrix *= 1 - evaporation

    def add_ph(self, better_path, better_path_len, Q):
        ph = Q / better_path_len
        for i in range(len(self.pheromone_matrix) - 1):
            self.pheromone_matrix[better_path[i]][better_path[i + 1]] += ph
            self.pheromone_matrix[better_path[i + 1]][better_path[i]] += ph
        self.pheromone_matrix[better_path[0]][better_path[len(self.pheromone_matrix) - 1]] += ph
        self.pheromone_matrix[better_path[len(self.pheromone_matrix) - 1]][better_path[0]] += ph

        for i in range(len(self.pheromone_matrix)):
            for j in range(len(self.pheromone_matrix)):
                if self.pheromone_matrix[i][j] > 1:
                    self.pheromone_matrix[i][j] = 1

    def setPH(self, ph):
        self.pheromone_matrix = np.full((len(self.cords), len(self.cords)), ph, dtype="double")

    def lenRandomPath(self):
        l = 0
        for i in range(len(self.distance_matrix) - 1):
            l += self.distance_matrix[i][i + 1]
        l += self.distance_matrix[1][len(self.distance_matrix) - 1]
        return l

    def visualize_best_path_2d(self, best_path):
        fig, ax = plt.subplots()

        xs = [n[0] for n in self.cords]
        ys = [n[1] for n in self.cords]
        ax.scatter(xs, ys)

        for i in range(len(best_path) - 1):
            x_start, y_start = self.cords[best_path[i]]
            x_end, y_end = self.cords[best_path[i + 1]]
            ax.plot([x_start, x_end], [y_start, y_end], 'g')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()


if __name__ == "__main__":
    import os

    g = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d100.txt')
    g.load(file_path)

    g.add_k_nearest_edges(20)
