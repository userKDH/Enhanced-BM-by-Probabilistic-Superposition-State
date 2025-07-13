"""
Simulated_annealing_max_cut_solver
"""

import random
import time
import numpy as np
import copy
import pandas as pd
import itertools


def read_max_cut_file(file_path):
    """
    Read the txt file of the max cut problem and return the number of nodes and weight dictionary

    Parameters:
        file_path (str): file path

    Return:
        num_nodes (int): number of nodes
        weights (dict): weight dictionary, format is {(i, j): w}
    """
    weights = {}
    with open(file_path, 'r') as file:
        # Read the first line to get the number of nodes and edges
        first_line = file.readline().strip()
        num_nodes, _ = map(int, first_line.split())

        # Read the remaining edge information
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                i, j, w = map(int, parts)
                # Make sure smaller nodes are first to avoid duplicate keys like (1,2) and (2,1)
                if i < j:
                    weights[(i, j)] = w
                else:
                    weights[(j, i)] = w
    return num_nodes, weights


def calculate_energy(solution, weights):
    energy = 0
    for (i, j), weight in weights.items():
        if solution[i] != solution[j]:  # If two nodes are located in different parts
            energy -= weight
    return energy


def propose_energy(current_solution, nodes_to_flip, weights):
    # Calculate the energy of the 16 energy levels of the four nodes involved
    all_states_energy = []
    new_solution = current_solution.copy()
    for bits in itertools.product([-1, 1], repeat=4):
        for i, node in enumerate(nodes_to_flip):
            new_solution[node] = bits[i]
        e = calculate_energy(new_solution, weights)
        all_states_energy.append(copy.deepcopy((new_solution, e)))
    return all_states_energy


def Simulated_annealing(graph, weights, initial_temp, cooling_rate, end_temp, num_iterations):
    """
    :param graph: list of nodes
    :param weights: {(i, j): weight}
    :param initial_temp
    :param cooling_rate (0 < cooling_rate < 1)
    :param num_iterations: Iteration count per temperature step
    :return: Optimal cut and associated energy
    """
    # Randomly initialize the solution, dividing nodes into two groups: {+1, -1}
    current_solution = {node: random.choice([1, -1]) for node in graph}
    # Calculate the energy of the initial solution
    current_energy = calculate_energy(current_solution, weights)
    best_solution = current_solution.copy()
    best_energy = current_energy
    temperature = initial_temp
    E_list = [current_energy]
    while temperature > end_temp:  # Terminate when temperature drops below threshold.
        for _ in range(num_iterations):
            # Randomly generate new solutions in the neighborhood of the current solution
            nodes_to_flip = random.sample(graph, 4)
            all_states_energy = propose_energy(current_solution, nodes_to_flip, weights)
            # Extract all energies and sort them (de-duplicate)
            all_energy = [energy for (solution, energy) in all_states_energy]
            unique_sorted_energy = sorted(set(all_energy))

            # Check whether the current system energy is the minimum value among the 16 energy levels
            if current_energy == unique_sorted_energy[0]:
                # The new solution proposed by perturbation is set as the second energy level state
                second_min_energy = unique_sorted_energy[1]
                # Collect all solutions with the second smallest energy (this step is to take into account possible energy degeneracy)
                second_min_states = [solution for (solution, energy) in all_states_energy if energy == second_min_energy]
                new_solution = random.choice(second_min_states).copy()
                new_energy = second_min_energy

                delta_E = current_energy - new_energy
                p_sample = 1 / np.exp(-delta_E / temperature)

                if np.random.rand() < p_sample:
                    current_solution = new_solution.copy()
                    current_energy = new_energy
                E_list.append(copy.copy(current_energy))

            else:
                # If the current system solution is not the lowest energy level of the 16 energy levels, it will definitely transfer to the optimal energy level.
                min_energy = unique_sorted_energy[0]
                min_states = [solution for (solution, energy) in all_states_energy if energy == min_energy]
                current_solution = random.choice(min_states).copy()
                current_energy = min_energy
                E_list.append(copy.copy(current_energy))
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
        # Temperature annealing
        temperature *= cooling_rate

    # You can export the solution sequence for subsequent analysis:
    df = pd.DataFrame(E_list)
    df.to_csv('.../Energy_evolution_list.csv', mode='w', header=False)
    return best_solution, -best_energy  # Returns max cut partition and the max cut value (negative energy is positive)


def main():
    """The following run instances and parameters can be changed"""
    # Please complete the input path
    num_nodes, weights = read_max_cut_file('.../gka1d.sparse.txt')
    start_time = time.time()
    graph = list(range(1, num_nodes + 1))
    initial_temp = 1500
    cooling_rate = 0.935
    num_iterations = 2000
    end_temp = 55

    solution, max_cut_value = Simulated_annealing(graph, weights, initial_temp, cooling_rate, end_temp, num_iterations)
    print("Max cut partition:", solution)
    print("Max cut value:", max_cut_value)

    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    main()