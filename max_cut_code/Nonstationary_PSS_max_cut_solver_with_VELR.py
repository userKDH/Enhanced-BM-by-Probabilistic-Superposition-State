"""
Nonstationary_PSS_max_cut_solver_with_VELR
"""

import numpy as np
import copy
import time
import random
import sys
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


def sigmoid(x: float, T: float) -> float:
    """Prevent overflow by constraining the input range"""
    try:
        exp_input = -x / T
        if exp_input > 709:  # the largest value that can be handled by floating-point numbers
            return 1e-5
        elif exp_input < -709:  # np.exp(-709) is very close to 0 and can be safely returned as 1.0
            return 1.0
        exp_value = np.exp(exp_input)
        return 1 / (1 + exp_value)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0.0


def find_eigenvector_for_eigenvalue_one(A, temperature, tol=1e-5, tol_1=0.5):
    eigenvalues, eigenvectors = np.linalg.eig(A)

    """Entanglement establishment verification"""
    # At extremely low temperatures, the probability transition matrix may have multiple eigenvectors with eigenvalue 1. For now, we will ignore this situation
    near_one_indices = np.where(np.abs(eigenvalues - 1) < tol)[0]
    t = 0
    for i in near_one_indices:
        a = np.sum(eigenvectors[:, i])
        # print(a)
        if abs(abs(a.real) - 1) < tol_1:
            t += 1
    if len(near_one_indices) > 1 and t > 1:
        print('Multiple stationary states')
        print('quit')
        print(A)
        print(eigenvalues)
        print(eigenvectors)
        print(temperature)
        sys.exit(1)

    """Find the stationary eigenvector and probabilistic error processing"""
    idx = np.argmin(np.abs(eigenvalues - 1))
    eigenvectors_near_one = eigenvectors[:, idx]
    eigenvectors_near_one = eigenvectors_near_one / np.sum(eigenvectors_near_one)
    # Ensure small negative values are set to zero (calculation error) and eliminate their influence
    eigenvectors_near_one = np.maximum(eigenvectors_near_one, 0)
    eigenvectors_near_one = eigenvectors_near_one / np.sum(eigenvectors_near_one)
    eigenvectors_near_one_real = np.real(eigenvectors_near_one)
    return eigenvectors_near_one_real


def two_spin_p_stable_distribution(weights, solution, spin_1, spin_2, T, nodes_to_flip, exchange_assign_list, energy_assign_list, assign_index):

    def get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list):
        """Determine whether it involves the energy level that VELR needs to exchange when calculating the stationary probabilities of two-node"""
        solution_node_judge = []
        for i in nodes_to_flip:
            solution_node_judge.append(solution[i])
        if solution_node_judge == exchange_assign_list[0]:
            return energy_assign_list[1]
        elif solution_node_judge == exchange_assign_list[1]:
            return energy_assign_list[0]
        else:
            return None

    """Calculate the stationary probabilities of the 2-node case"""
    """Calculate the Gibbs probability of Node 2 values (under the condition that Node 1 takes values 1 or 0)"""
    p_list_1 = []
    for k in range(-1, 2, 2):
        solution[spin_1] = k
        solution[spin_2] = -1
        if assign_index:
            if get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list) is None:
                old_energy = calculate_energy(solution, weights)
            else:
                old_energy = get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list)
        else:
            old_energy = calculate_energy(solution, weights)

        solution[spin_2] = 1
        if assign_index:
            if get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list) is None:
                new_energy = calculate_energy(solution, weights)
            else:
                new_energy = get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list)
        else:
            new_energy = calculate_energy(solution, weights)

        V = old_energy - new_energy
        p_list_1.append(sigmoid(V, T))  # p_list_1 = [q, p]
    q = p_list_1[0]
    p = p_list_1[1]

    """Calculate the Gibbs probability of Node 1 values (under the condition that Node 2 takes values 1 or 0)"""
    p_list_2 = []
    for k in range(-1, 2, 2):
        solution[spin_2] = k
        solution[spin_1] = -1
        if assign_index:
            if get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list) is None:
                old_energy = calculate_energy(solution, weights)
            else:
                old_energy = get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list)
        else:
            old_energy = calculate_energy(solution, weights)

        solution[spin_1] = 1
        if assign_index:
            if get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list) is None:
                new_energy = calculate_energy(solution, weights)
            else:
                new_energy = get_assign_energy(solution, nodes_to_flip, exchange_assign_list, energy_assign_list)
        else:
            new_energy = calculate_energy(solution, weights)

        V = old_energy - new_energy
        p_list_2.append(sigmoid(V, T))  # p_list_1 = [Q, P]
    Q = p_list_2[0]
    P = p_list_2[1]

    """"Calculate the stationary probabilities of the 2-node case"""
    a_x_1 = (1 - P) * p + (1 - Q) * (1 - p)
    b_x = P * q + Q * (1 - q)
    c_x = b_x / (a_x_1 + b_x)

    a_y_1 = (1 - p) * P + (1 - q) * (1 - P)
    b_y = p * Q + q * (1 - Q)
    c_y = b_y / (a_y_1 + b_y)
    c_list = [c_x, c_y]
    return c_list


def calculate_energy(solution, weights):
    energy = 0
    for (i, j), weight in weights.items():
        if solution[i] != solution[j]:  # If two nodes are located in different parts
            energy -= weight
    return energy


def node_choose(node_list):
    if node_list == [-1, -1]:
        return 0
    elif node_list == [-1, 1]:
        return 1
    elif node_list == [1, -1]:
        return 2
    else:
        return 3


def matrixPow(matrix, n):
    if n == 1:
        return matrix
    else:
        return np.matmul(matrix, matrixPow(matrix, n - 1))


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


def generate_VELR_combinations(input_list):
    # Extract the first and last parts of the node list
    first_two = input_list[:2]
    last_two = input_list[2:]

    # Generate all possible combinations of the first part (different from the input)
    possible_first = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            if [a, b] != first_two:
                possible_first.append([a, b])

    # Generate all possible combinations of the last part (different from the input)
    possible_last = []
    for c in [-1, 1]:
        for d in [-1, 1]:
            if [c, d] != last_two:
                possible_last.append([c, d])

    # Combine the first and last parts
    result = []
    for pf in possible_first:
        for pl in possible_last:
            result.append(pf + pl)

    return result


def four_node_PSS(graph, weights, initial_temp, cooling_rate, end_temp, num_iterations, n):
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

                solution_node_old = []
                solution_node_new = []
                for node in nodes_to_flip:
                    solution_node_old.append(current_solution[node])
                    solution_node_new.append(new_solution[node])

                solution_node_new_assign = []
                exchange_assign_list = []
                energy_assign_list = []
                VELR_assign = 0
                # Determine whether the optimal and suboptimal energy levels satisfy the phase space separation requirements
                if (solution_node_old[:2] == solution_node_new[:2]) or (solution_node_old[2:] == solution_node_new[2:]):
                    # If not satisfied
                    VELR_assign = 1  # the VELR strategy is applied to virtually reconstruct the phase space energy
                    solution_node_assign_list = generate_VELR_combinations(solution_node_old)  # Generate a 4-node value list separated from the optimal energy level state
                    solution_node_new_assign = random.choice(solution_node_assign_list).copy()
                    solution_assign = current_solution.copy()
                    for i, node in enumerate(nodes_to_flip):
                        solution_assign[node] = solution_node_new_assign[i]
                    assign_energy = calculate_energy(solution_assign, weights)
                    # The following are the energy levels and solutions that meet the requirements of VELR exchange:
                    energy_assign_list = [new_energy, assign_energy]
                    exchange_assign_list = [solution_node_new, solution_node_new_assign]

                solution = current_solution.copy()
                transfer_matrix_0 = np.zeros((4, 4), dtype=float)
                transfer_matrix_1 = np.zeros((4, 4), dtype=float)
                spin_1 = nodes_to_flip[0]
                spin_2 = nodes_to_flip[1]
                ip = 0
                for m in range(-1, 2, 2):
                    for n in range(-1, 2, 2):
                        solution[nodes_to_flip[2]] = m
                        solution[nodes_to_flip[3]] = n
                        c = two_spin_p_stable_distribution(weights, solution, spin_1, spin_2, temperature, nodes_to_flip, exchange_assign_list, energy_assign_list, VELR_assign)
                        transfer_matrix_0[0, ip] = (1 - c[0]) * (1 - c[1])
                        transfer_matrix_0[1, ip] = (1 - c[0]) * c[1]
                        transfer_matrix_0[2, ip] = c[0] * (1 - c[1])
                        transfer_matrix_0[3, ip] = c[0] * c[1]
                        ip += 1
                spin_1 = nodes_to_flip[2]
                spin_2 = nodes_to_flip[3]
                ip = 0
                for m in range(-1, 2, 2):
                    for n in range(-1, 2, 2):
                        solution[nodes_to_flip[0]] = m
                        solution[nodes_to_flip[1]] = n
                        c = two_spin_p_stable_distribution(weights, solution, spin_1, spin_2, temperature, nodes_to_flip, exchange_assign_list, energy_assign_list, VELR_assign)
                        transfer_matrix_1[0, ip] = (1 - c[0]) * (1 - c[1])
                        transfer_matrix_1[1, ip] = (1 - c[0]) * c[1]
                        transfer_matrix_1[2, ip] = c[0] * (1 - c[1])
                        transfer_matrix_1[3, ip] = c[0] * c[1]
                        ip += 1
                """Obtain the self-transition matrix for each group of nodes"""
                transfer_matrix_12 = transfer_matrix_0.dot(transfer_matrix_1)
                transfer_matrix_34 = transfer_matrix_1.dot(transfer_matrix_0)
                """Calculate the non-stationary transfer matrix"""
                transfer_matrix_12_f = matrixPow(transfer_matrix_12, n)
                transfer_matrix_34_f = matrixPow(transfer_matrix_34, n)
                # Stationary PSS version:
                # transfer_matrix_12_f = find_eigenvector_for_eigenvalue_one(transfer_matrix_12, T)
                # transfer_matrix_34_f = find_eigenvector_for_eigenvalue_one(transfer_matrix_34, T)
                # if VELR_assign == 1:
                #     p1 = transfer_matrix_12_f[node_choose([solution_node_new_assign[0], solution_node_new_assign[1]])] * transfer_matrix_34_f[node_choose([solution_node_new_assign[2], solution_node_new_assign[3]])]
                # else:
                #     p1 = transfer_matrix_12_f[node_choose([solution_node_new[0], solution_node_new[1]])] * transfer_matrix_34_f[node_choose([solution_node_new[2], solution_node_new[3]])]
                # p2 = transfer_matrix_12_f[node_choose([solution_node_old[0], solution_node_old[1]])] * transfer_matrix_34_f[node_choose([solution_node_old[2], solution_node_old[3]])]

                if VELR_assign == 1:
                    p1 = transfer_matrix_12_f[node_choose([solution_node_new_assign[0], solution_node_new_assign[1]]), node_choose([solution_node_old[0], solution_node_old[1]])] * transfer_matrix_34_f[node_choose([solution_node_new_assign[2], solution_node_new_assign[3]]), node_choose([solution_node_old[2], solution_node_old[3]])]
                else:
                    p1 = transfer_matrix_12_f[node_choose([solution_node_new[0], solution_node_new[1]]), node_choose([solution_node_old[0], solution_node_old[1]])] * transfer_matrix_34_f[node_choose([solution_node_new[2], solution_node_new[3]]), node_choose([solution_node_old[2], solution_node_old[3]])]
                p2 = transfer_matrix_12_f[node_choose([solution_node_old[0], solution_node_old[1]]), node_choose([solution_node_old[0], solution_node_old[1]])] * transfer_matrix_34_f[node_choose([solution_node_old[2], solution_node_old[3]]), node_choose([solution_node_old[2], solution_node_old[3]])]

                p_sample = p1 / (p1 + p2)  # Renormalize PSS probabilities of the two solutions
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
    end_temp = 55
    num_iterations = 2000
    n = 1  # Transfer matrix iteration number
    solution, max_cut_value = four_node_PSS(graph, weights, initial_temp, cooling_rate, end_temp, num_iterations, n)
    print("Max cut partition:", solution)
    print("Max cut value:", max_cut_value)

    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    main()
