"""
Simulated_annealing_CVRP_solver
"""

import random
import time
import numpy as np
import copy
import pandas as pd
import itertools


def __sample(p: float) -> int:
    """
    :param p: probability
    :return: 0/1
    """
    value = np.random.choice([0, 1], p=[1 - p, p])
    return value


def distance_2D(a: float, b: float, c: float, d: float):
    Dis: int = round(((a-c)**2 + (b-d)**2)**0.5)
    return Dis


def extract_node_and_demand_info(file_content):
    """Load the coordinates of the depot and customer, and the demands of the customers"""
    # Read the file content and split it by line
    lines = file_content.split('\n')

    # Initialize the coordinate list and demand list
    coordinates = []
    demands = []

    # Flag to identify the current section being read
    in_node_coord_section = False
    in_demand_section = False

    for line in lines:
        line = line.strip()

        # Check if we have entered NODE_COORD_SECTION
        if line == "NODE_COORD_SECTION":
            in_node_coord_section = True
            in_demand_section = False
            continue

        # Check if we have entered DEMAND_SECTION
        if line == "DEMAND_SECTION":
            in_node_coord_section = False
            in_demand_section = True
            continue

        # Check if we have ended the section
        if line in ["DEPOT_SECTION", "EOF"]:
            in_node_coord_section = False
            in_demand_section = False
            continue

        # Read coordinate information
        if in_node_coord_section and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates.append([node_id, x, y])

        # Read demand information
        if in_demand_section and line:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands.append([node_id, demand])

    return coordinates, demands


def start_solution(City_nums, Capacity, demands, vehicle):
    """Generate a reasonable solution sequence that meets the vehicle number and capacity requirements"""
    while True:
        sequence = random.sample(range(1, City_nums), City_nums - 1)
        sequence = [0] + sequence + [0]
        # Generate all combinations of m positions out of k (1-based between elements)
        positions = range(1, len(sequence))  # positions are after original_list[i-1], before original_list[i]
        all_combinations = itertools.combinations(positions, vehicle - 1)

        result = []
        for combo in all_combinations:
            new_list = sequence.copy()
            # Insert zeros from the end to avoid shifting issues
            for pos in sorted(combo, reverse=True):
                new_list.insert(pos, 0)
            result.append(new_list)
        valid_result = []
        for new_list in result:
            if valid_judge(new_list, demands, Capacity):
                valid_result.append(new_list)
        if valid_result:
            return random.choice(valid_result)


def valid_judge(solution, demands, Capacity):
    """Calculate the total cargo volume of each vehicle based on the input solution sequence and check whether it exceeds the capacity"""
    def energy_func(subset, demands):
        # calculate the total cargo volume
        E = 0
        for s in range(len(subset)):
            E += demands[subset[s]][1]
        return E
    zero_indices = [i for i, x in enumerate(solution) if x == 0]

    # Split the list based on the position of zeros
    sublists = []
    for i in range(len(zero_indices) - 1):
        start = zero_indices[i]
        end = zero_indices[i + 1]
        sublist = solution[start:end + 1]
        sublists.append(sublist)
    for sublist in sublists:
        d = energy_func(sublist, demands)
        if d > Capacity:
            return False
    return True


def solution_repair(Capacity, demands, vehicle, solution):
    """Repair the input solution so that it meets the vehicle capacity requirements"""
    if valid_judge(solution, demands, Capacity):
        return solution

    filtered_solution = [x for x in solution if x != 0]
    filtered_solution = [0] + filtered_solution + [0]

    positions = range(1, len(filtered_solution))  # positions are after original_list[i-1], before original_list[i]
    all_combinations = itertools.combinations(positions, vehicle-1)

    result = []
    for combo in all_combinations:
        new_list = filtered_solution.copy()
        # Insert zeros from the end to avoid shifting issues
        for pos in sorted(combo, reverse=True):
            new_list.insert(pos, 0)
        result.append(new_list)

    valid_result = []
    for new_list in result:
        if valid_judge(new_list, demands, Capacity):
            valid_result.append(new_list)

    if not valid_result:
        return None
    return random.choice(valid_result)


def main():
    """The following run instances and parameters can be changed"""
    np.set_printoptions(threshold=np.inf)
    start_time = time.time()
    with open('.../P-n22-k8.vrp.txt', 'r') as file:
        file_content = file.read()
    coordinates, demands = extract_node_and_demand_info(
        file_content)  # Load the coordinates of the depot and customer, and the demands of the customers
    City_nums = 22  # The total number of depots and customers
    D_matrix = np.zeros((City_nums, City_nums))  # Create the distance matrix
    for i in range(City_nums):
        for j in range(City_nums):
            D_matrix[i, j] = distance_2D(float(coordinates[i][1]), float(coordinates[i][2]), float(coordinates[j][1]), float(coordinates[j][2]))
    V = 8  # number of vehicle
    Capacity = 3000  # the capacity limit of vehicle
    initial_solution = start_solution(City_nums, Capacity, demands, V)  # generate the initial solution sequence

    iteration = 2000  # the number of iterations per temperature
    K: float = 400  # penalty factor K
    T = 90  # initial temperature
    cooling_rate = 0.95
    end_temp = 25

    """Total distance(energy) calculation function"""
    def path_func(Neural, D_matrix):
        E = 0
        for i in range(Neural.shape[0] - 1):
            middle = Neural[i + 1, :].reshape((City_nums, 1))
            E += (Neural[i, :] @ D_matrix @ middle).item()
        return E

    # Convert the initial solution sequence into matrix form
    Neural = np.zeros((len(initial_solution), City_nums), dtype=float)
    j = 0
    for i in range(len(initial_solution)):
        Neural[j, initial_solution[i]] = 1
        j += 1

    E_initial = path_func(Neural, D_matrix)  # Calculate the energy of the initial solution
    E_min = E_initial
    solution_min = initial_solution
    E_restrain = [E_initial]  # Create a solution sequence list

    solution = initial_solution
    while T > end_temp:
        for it in range(iteration):
            E_now = path_func(Neural, D_matrix)
            Neural_before = copy.deepcopy(Neural)

            while True:
                non_zero_indices = [i for i, x in enumerate(solution) if x != 0]
                r_1, r_2 = random.sample(non_zero_indices, 2)
                new_solution = solution.copy()
                new_solution[r_1], new_solution[r_2] = new_solution[r_2], new_solution[
                    r_1]  # Perturbation: Exchange the access order of any two customers of the current solution
                # The new solution after the exchange may violate the vehicle capacity requirements and need to be repaired
                new_solution_repaired = solution_repair(Capacity, demands, V, new_solution)
                if new_solution_repaired is not None:
                    break
            # Get the matrix form and energy of the repair solution
            Neural_repaired = np.zeros((len(new_solution_repaired), City_nums), dtype=float)
            j = 0
            for i in range(len(new_solution_repaired)):
                Neural_repaired[j, new_solution_repaired[i]] = 1
                j += 1
            E_repaired = path_func(Neural_repaired, D_matrix)

            # If the energy of the repair solution decreases, then it must be accepted
            if E_repaired <= E_now:
                E_restrain.append(E_repaired)
                Neural = copy.deepcopy(Neural_repaired)
                solution = copy.deepcopy(new_solution_repaired)
                if E_repaired < E_min:
                    E_min = E_repaired
                    solution_min = copy.deepcopy(new_solution_repaired)
            else:
                p1_r = np.exp((E_now - E_repaired) / T)
                value = __sample(p1_r)
                if value == 1:
                    E_restrain.append(E_repaired)
                    Neural = copy.deepcopy(Neural_repaired)
                    solution = copy.deepcopy(new_solution_repaired)
                    if E_repaired < E_min:
                        E_min = E_repaired
                        solution_min = copy.deepcopy(new_solution_repaired)
                else:
                    E_restrain.append(E_now)
                    Neural = copy.deepcopy(Neural_before)
        T *= cooling_rate

    # You can export the solution sequence for subsequent analysis:
    df = pd.DataFrame(E_restrain)
    df.to_csv('.../Energy_evolution_list.csv', mode='w', header=False)
    print(E_min)
    print(solution_min)
    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    main()

