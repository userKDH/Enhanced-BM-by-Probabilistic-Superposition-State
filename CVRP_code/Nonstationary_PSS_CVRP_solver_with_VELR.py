"""
Nonstationary_PSS_CVRP_solver_with_VELR
"""
import random
import sys
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


def distance_2D(a: float, b: float, c: float, d: float):
    Dis: int = round(((a-c)**2 + (b-d)**2)**0.5)
    return Dis


def two_spin_p_stable_distribution(City_nums: int, D_matrix: np.ndarray, Neural: np.ndarray, spin_1: list, spin_2: list, T: float, K:float, repair_index, E_repaired):
    def path_func(Neural, D_matrix):
        E = 0
        for i in range(Neural.shape[0] - 1):
            middle = Neural[i + 1, :].reshape((City_nums, 1))
            E += (Neural[i, :] @ D_matrix @ middle).item()
        return E

    def non_valid_energy(Neural, K):
        """Calculate the energy of the penalty term"""
        # Step 1: Remove rows where the first element is 1
        new_array = Neural[Neural[:, 0] != 1]

        # Step 2: Remove the first column
        final_array = new_array[:, 1:]

        # Step 3: Calculate row energy
        E_row = 0
        for row in final_array:
            count_ones = np.count_nonzero(row == 1)
            if count_ones == 0 or count_ones == 2:
                E_row += K

        # Step 4: Calculate column energy
        E_col = 0
        for j in range(final_array.shape[1]):
            column = final_array[:, j]
            count_ones = np.count_nonzero(column == 1)
            if count_ones == 0 or count_ones == 2:
                E_col += K

        # Step 5: Total energy
        E_total = E_row + E_col
        return E_total

    i = spin_1[0]
    j = spin_1[1]
    i1 = spin_2[0]
    j1 = spin_2[1]
    p_list_1 = []
    for k in range(2):
        Neural[i, j] = k
        Neural[i1, j1] = 0
        old_energy = path_func(Neural, D_matrix) + non_valid_energy(Neural, K)

        Neural[i1, j1] = 1
        #  Replace the new solution with the repaired solution
        if repair_index and [Neural[i, j], Neural[i1, j1]] == [0, 1]:
            new_energy = E_repaired
        else:
            new_energy = path_func(Neural, D_matrix) + non_valid_energy(Neural, K)
        V = old_energy - new_energy
        p_list_1.append(sigmoid(V, T))  # p_list_1 = [q, p]
    q = p_list_1[0]
    p = p_list_1[1]

    """Calculate the Gibbs probability of Node 1 values (under the condition that Node 2 takes values 1 or 0)"""
    p_list_2 = []
    for k in range(2):
        Neural[i1, j1] = k
        Neural[i, j] = 0
        if repair_index and [Neural[i, j], Neural[i1, j1]] == [0, 1]:
            old_energy = E_repaired
        else:
            old_energy = path_func(Neural, D_matrix) + non_valid_energy(Neural, K)

        Neural[i, j] = 1
        new_energy = path_func(Neural, D_matrix) + non_valid_energy(Neural, K)
        V = old_energy - new_energy
        p_list_2.append(sigmoid(V, T))  # p_list_1 = [Q, P]
    Q = p_list_2[0]
    P = p_list_2[1]


    a_x_1 = (1 - P) * p + (1 - Q) * (1 - p)
    b_x = P * q + Q * (1 - q)
    c_x = b_x / (a_x_1 + b_x)

    a_y_1 = (1 - p) * P + (1 - q) * (1 - P)
    b_y = p * Q + q * (1 - Q)
    c_y = b_y / (a_y_1 + b_y)
    c_list = [c_x, c_y]
    return c_list


def matrixPow(matrix, n):
    if n == 1:
        return matrix
    else:
        return np.matmul(matrix, matrixPow(matrix, n - 1))


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
    coordinates, demands = extract_node_and_demand_info(file_content)  # Load the coordinates of the depot and customer, and the demands of the customers
    City_nums = 22  # The total number of depots and customers
    D_matrix = np.zeros((City_nums, City_nums))   # Create the distance matrix
    for i in range(City_nums):
        for j in range(City_nums):
            D_matrix[i, j] = distance_2D(float(coordinates[i][1]), float(coordinates[i][2]), float(coordinates[j][1]), float(coordinates[j][2]))
    V = 8   # number of vehicle
    Capacity = 3000  # the capacity limit of vehicle
    initial_solution = start_solution(City_nums, Capacity, demands, V)  # generate the initial solution sequence

    iteration = 2000  # the number of iterations per temperature
    K: float = 400  # penalty factor K
    T = 90  # initial temperature
    cooling_rate = 0.95
    end_temp = 25
    n_transfer = 800  # transfer matrix iteration number

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

    co_1 = 0
    co_2 = 0
    solution = initial_solution
    while T > end_temp:
        for it in range(iteration):
            E_now = path_func(Neural, D_matrix)
            Neural_before = copy.deepcopy(Neural)

            while True:
                non_zero_indices = [i for i, x in enumerate(solution) if x != 0]
                r_1, r_2 = random.sample(non_zero_indices, 2)
                new_solution = solution.copy()
                new_solution[r_1], new_solution[r_2] = new_solution[r_2], new_solution[r_1]  # Perturbation: Exchange the access order of any two customers of the current solution
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
                # Select 4 nodes of PSS
                spin = []
                for j in range(City_nums):
                    if Neural[r_1, j] == 1:
                        co_1 = copy.deepcopy(j)
                for j in range(City_nums):
                    if Neural[r_2, j] == 1:
                        co_2 = copy.deepcopy(j)
                spin.append(copy.deepcopy([r_1, co_1]))
                spin.append(copy.deepcopy([r_2, co_1]))
                spin.append(copy.deepcopy([r_2, co_2]))
                spin.append(copy.deepcopy([r_1, co_2]))

                transfer_matrix_0 = np.zeros((4, 4), dtype=float)
                transfer_matrix_1 = np.zeros((4, 4), dtype=float)
                spin_1 = spin[0]
                spin_2 = spin[1]
                ip = 0
                for m in range(2):
                    for n in range(2):
                        repair_index = False
                        Neural[spin[2][0], spin[2][1]] = m
                        Neural[spin[3][0], spin[3][1]] = n
                        if [m, n] == [0, 1]:
                            repair_index = True
                        c = two_spin_p_stable_distribution(City_nums, D_matrix, Neural, spin_1, spin_2, T, K, repair_index, E_repaired)
                        transfer_matrix_0[0, ip] = (1 - c[0]) * (1 - c[1])
                        transfer_matrix_0[1, ip] = (1 - c[0]) * c[1]
                        transfer_matrix_0[2, ip] = c[0] * (1 - c[1])
                        transfer_matrix_0[3, ip] = c[0] * c[1]
                        ip += 1
                spin_1 = spin[2]
                spin_2 = spin[3]
                ip = 0
                for m in range(2):
                    for n in range(2):
                        repair_index = False
                        Neural[spin[0][0], spin[0][1]] = m
                        Neural[spin[1][0], spin[1][1]] = n
                        if [m, n] == [0, 1]:
                            repair_index = True
                        c = two_spin_p_stable_distribution(City_nums, D_matrix, Neural, spin_1, spin_2, T, K, repair_index, E_repaired)
                        transfer_matrix_1[0, ip] = (1 - c[0]) * (1 - c[1])
                        transfer_matrix_1[1, ip] = (1 - c[0]) * c[1]
                        transfer_matrix_1[2, ip] = c[0] * (1 - c[1])
                        transfer_matrix_1[3, ip] = c[0] * c[1]
                        ip += 1

                transfer_matrix_12 = transfer_matrix_0.dot(transfer_matrix_1)
                transfer_matrix_34 = transfer_matrix_1.dot(transfer_matrix_0)
                transfer_matrix_12_f = matrixPow(transfer_matrix_12, n_transfer)
                transfer_matrix_34_f = matrixPow(transfer_matrix_34, n_transfer)
                p1 = transfer_matrix_12_f[1, 2] * transfer_matrix_34_f[1, 2]
                p2 = transfer_matrix_12_f[2, 2] * transfer_matrix_34_f[2, 2]
                # Stationary PSS version:
                # transfer_matrix_12_f = find_eigenvector_for_eigenvalue_one(transfer_matrix_12, T)
                # transfer_matrix_34_f = find_eigenvector_for_eigenvalue_one(transfer_matrix_34, T)
                # p1 = transfer_matrix_12_f[1] * transfer_matrix_34_f[1]
                # p2 = transfer_matrix_12_f[2] * transfer_matrix_34_f[2]
                p1_r = p1 / (p1 + p2)
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

