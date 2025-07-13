"""
Nonstationary_PSS_sampling for TSP
"""

import copy
import numpy as np
import time
import random
import pandas as pd


class TSP:
    def __init__(self, instance_name, city_nums, file_path_input_distance_matrix, file_path_output_energy_list):
        self.instance_name = instance_name
        self.city_nums = city_nums
        self.file_path_input_distance_matrix = file_path_input_distance_matrix
        self.file_path_output_energy_list = file_path_output_energy_list
        self.K = None
        self.T0 = None
        self.alpha = None
        self.n_transfer = None
        self.d_matrix = None
        self.load_instance()
        self.load_parameters()

    def load_parameters(self):
        """Load parameter settings based on the instance name, including the penalty factor K, the initial temperature for exponential annealing T0, the exponential factor alpha and transition matrix iteration number n_transfer"""
        parameter_mapping = {
            "gr17": {"K": 500, "T0": 100, "alpha": 0.976, "n_transfer": 300}, "gr24": {"K": 150, "T0": 100, "alpha": 0.955, "n_transfer": 5},
            "fri26": {"K": 300, "T0": 80, "alpha": 0.968, "n_transfer": 100}, "swiss42": {"K": 150, "T0": 50, "alpha": 0.968, "n_transfer": 50},
            "gr48": {"K": 600, "T0": 200, "alpha": 0.972, "n_transfer": 50}, "att48": {"K": 2000, "T0": 1000, "alpha": 0.963, "n_transfer": 300},
            "hk48": {"K": 3000, "T0": 1000, "alpha": 0.968, "n_transfer": 300},  "berlin52": {"K": 2000, "T0": 800, "alpha": 0.966, "n_transfer": 300},
            "st70": {"K": 70, "T0": 50, "alpha": 0.955, "n_transfer": 30}, "eil76": {"K": 100, "T0": 50, "alpha": 0.956, "n_transfer": 300},
            "rd100": {"K": 300, "T0": 300, "alpha": 0.964, "n_transfer": 1}, "kroA100": {"K": 2000, "T0": 2000, "alpha": 0.955, "n_transfer": 5},
            "kroB100": {"K": 2000, "T0": 2000, "alpha": 0.955, "n_transfer": 5}, "kroC100": {"K": 2000, "T0": 2000, "alpha": 0.955, "n_transfer": 5},
            "pr107": {"K": 3000, "T0": 4000, "alpha": 0.958, "n_transfer": 1}, "gr120": {"K": 300, "T0": 300, "alpha": 0.964, "n_transfer": 1},
            "ch130": {"K": 200, "T0": 200, "alpha": 0.972, "n_transfer": 1}, "ch150": {"K": 200, "T0": 200, "alpha": 0.972, "n_transfer": 1},
            "kroA150": {"K": 2000, "T0": 2000, "alpha": 0.955, "n_transfer": 3}, "kroA200": {"K": 2000, "T0": 1000, "alpha": 0.962, "n_transfer": 5}
            # You can change the running parameters or add other instances
        }

        if self.instance_name in parameter_mapping:
            params = parameter_mapping[self.instance_name]
            self.K = params["K"]
            self.T0 = params["T0"]
            self.alpha = params["alpha"]
            self.n_transfer = params["n_transfer"]
        else:
            raise ValueError(f"No parameter settings available for instance: {self.instance_name}")
        print(f"Parameters loaded for {self.instance_name}: penalty_factor: K={self.K}, initial_temperature: T0={self.T0}, exponential_annealing_factor: alpha={self.alpha}, transfer_matrix_iteration_number: n_transfer={self.n_transfer} (adjustable)")

    def load_instance(self):
        """Load the specified TSP instance and compute the corresponding distance matrix"""
        if self.instance_name == "swiss42":
            file_path = f"{self.file_path_input_distance_matrix}/{self.instance_name}-matrix-DIY.txt"
            self.d_matrix = np.array(self.read_distance_type_MATRIX(file_path)).reshape((self.city_nums, self.city_nums))
        elif self.instance_name in ["berlin52", "st70", "eil76", "rd100", "kroA100", "kroB100", "kroC100", "pr107", "ch130", "ch150", "kroA150", "kroA200"]:
            file_path = f"{self.file_path_input_distance_matrix}/{self.instance_name}-matrix-DIY.txt"
            d_list = self.read_distance_type_EUC_2D(file_path)
            self.d_matrix = np.zeros((self.city_nums, self.city_nums))
            for i in range(self.city_nums):
                for j in range(self.city_nums):
                    self.d_matrix[i, j] = self.distance_calculate_EUC_2D(
                        float(d_list[i][1]), float(d_list[i][2]),
                        float(d_list[j][1]), float(d_list[j][2])
                    )
        elif self.instance_name in ["gr17", "gr24", "fri26", "gr48", "hk48", "gr120"]:
            file_path = f"{self.file_path_input_distance_matrix}/{self.instance_name}-matrix-DIY.txt"
            d_matrix_0 = self.read_distance_type_MATRIX(file_path)
            self.d_matrix = self.process_distance_type_MATRIX_LOWER_DIAG_ROW(d_matrix_0)
        elif self.instance_name == "att48":
            file_path = f"{self.file_path_input_distance_matrix}/{self.instance_name}-matrix-DIY.txt"
            d_list = self.read_distance_type_EUC_2D(file_path)
            self.d_matrix = np.zeros((self.city_nums, self.city_nums))
            for i in range(self.city_nums):
                for j in range(self.city_nums):
                    self.d_matrix[i, j] = self.distance_calculate_ATT(
                        float(d_list[i][1]), float(d_list[i][2]),
                        float(d_list[j][1]), float(d_list[j][2])
                    )
        else:
            raise ValueError(f"Unsupported instance: {self.instance_name}")
        print(f"Distance matrix Loaded for {self.instance_name} with {self.city_nums} cities")
        print('')

    @staticmethod
    def read_distance_type_MATRIX(file_path):
        """Read the distance list from the file"""
        file = open(file_path, 'r')
        list_row = file.readlines()
        list_source = []
        for i in range(len(list_row)):
            column_string_list = list_row[i].strip().split()
            list_source += column_string_list
        for i in range(len(list_source)):
            list_source[i] = int(list_source[i])
        file.close()
        return list_source

    @staticmethod
    def read_distance_type_EUC_2D(file_path):
        """Read the distance list from the file"""
        file = open(file_path, 'r')
        list_row = file.readlines()
        list_source = []
        for i in range(len(list_row)):
            column_string_list = list_row[i].strip().split()
            list_source.append(column_string_list)
        file.close()
        return list_source

    @staticmethod
    def distance_calculate_ATT(a, b, c, d):
        """Calculate distances in the ATT distance matrix"""
        x_d = a - c
        y_d = b - d
        rij = ((x_d ** 2 + y_d ** 2) / 10.0) ** 0.5
        tij = round(rij)
        if tij < rij:
            dij = tij + 1
        else:
            dij = tij
        return dij

    @staticmethod
    def distance_calculate_EUC_2D(a: float, b: float, c: float, d: float):
        Dis: int = round(((a - c) ** 2 + (b - d) ** 2) ** 0.5)
        return Dis

    def process_distance_type_MATRIX_LOWER_DIAG_ROW(self, d_matrix_0):
        """Calculate distances in the LOWER_DIAG_ROW distance matrix"""
        d_matrix = np.zeros((self.city_nums, self.city_nums))
        d1 = d2 = 0
        for i in range(len(d_matrix_0)):
            if d_matrix_0[i] == 0:
                d1 += 1
                d2 = -1
            else:
                d2 += 1
                d_matrix[d1, d2] = d_matrix_0[i]
        for i in range(self.city_nums):
            for j in range(self.city_nums):
                if i < j:
                    d_matrix[i, j] = d_matrix[j, i]
        return d_matrix

    @staticmethod
    def exponential_annealing(T0, alpha, k):
        """Exponential annealing"""
        return T0 * (alpha ** k)

    @staticmethod
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

    @staticmethod
    def matrixPow(matrix, n):
        """Matrix power multiplication"""
        if n == 1:
            return matrix
        else:
            return np.matmul(matrix, TSP.matrixPow(matrix, n - 1))

    def two_spin_p_stationary_distribution(self, Neuron: np.ndarray, spin_1: list, spin_2: list, T: float):
        """"Calculate the stationary probabilities of the 2-node case"""
        City_nums = self.city_nums
        D_matrix = self.d_matrix
        K = self.K

        i = spin_1[0]
        j = spin_1[1]
        i1 = spin_2[0]
        j1 = spin_2[1]
        """Calculate the Gibbs probability of Node 2 values (under the condition that Node 1 takes values 1 or 0)"""
        p_list_1 = []
        for k in range(2):
            Neuron[i, j] = k
            V1 = 0
            if i1 == (City_nums - 1):
                V1 += np.dot(Neuron[i1 - 1, :], D_matrix[:, j1])
                V1 += np.dot(Neuron[0, :], D_matrix[:, j1])
            else:
                V1 += np.dot(Neuron[i1 + 1, :], D_matrix[:, j1])
                V1 += np.dot(Neuron[i1 - 1, :], D_matrix[:, j1])
            V2 = 0
            for m in range(City_nums):
                if m == j1:
                    continue
                V2 += 2 * K * Neuron[i1, m]
            for n in range(City_nums):
                if n == i1:
                    continue
                V2 += 2 * K * Neuron[n, j1]
            V3 = -2 * K
            V = -(V1 + V2 + V3)
            p_list_1.append(self.sigmoid(V, T))  # p_list_1 = [q, p]
        q = p_list_1[0]
        p = p_list_1[1]
        """Calculate the Gibbs probability of Node 1 values (under the condition that Node 2 takes values 1 or 0)"""
        p_list_2 = []
        for k in range(2):
            Neuron[i1, j1] = k
            V1 = 0
            if i == (City_nums - 1):
                V1 += np.dot(Neuron[i - 1, :], D_matrix[:, j])
                V1 += np.dot(Neuron[0, :], D_matrix[:, j])
            else:
                V1 += np.dot(Neuron[i + 1, :], D_matrix[:, j])
                V1 += np.dot(Neuron[i - 1, :], D_matrix[:, j])
            V2 = 0
            for m in range(City_nums):
                if m == j:
                    continue
                V2 += 2 * K * Neuron[i, m]
            for n in range(City_nums):
                if n == i:
                    continue
                V2 += 2 * K * Neuron[n, j]
            V3 = -2 * K
            V = -(V1 + V2 + V3)
            p_list_2.append(self.sigmoid(V, T))  # p_list_2 = [Q, P]
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

    def Nonstationary_PSS_TSP_solver(self):
        City_nums = self.city_nums
        D_matrix = self.d_matrix
        T0 = self.T0
        alpha = self.alpha
        n_transfer = self.n_transfer
        Neuron = np.zeros((City_nums, City_nums), dtype=float)
        Energy_list = []
        annealing_T_number = 50  # The number of annealing temperatures (adjustable)
        print('The number of annealing temperatures: %s (adjustable)' % annealing_T_number)
        exponential_temps = [self.exponential_annealing(T0, alpha, k) for k in range(annealing_T_number)]  # Exponential annealing
        iteration = 100  # The number of iterations at each temperature (adjustable)
        print('The number of iterations at each temperature: %s (adjustable)' % iteration)
        print('')

        Random_initial = np.random.permutation(City_nums)
        j = 0
        for i in Random_initial:
            Neuron[j, i] = 1
            j += 1

        E_initial = 0
        for j in range(City_nums):
            if j == (City_nums - 1):
                middle = Neuron[0, :].reshape((City_nums, 1))
                E_initial += (Neuron[j, :] @ D_matrix @ middle).item()
            else:
                middle = Neuron[j + 1, :].reshape((City_nums, 1))
                E_initial += (Neuron[j, :] @ D_matrix @ middle).item()
        Energy_list.append(E_initial)

        Neuron_min = copy.deepcopy(Neuron)
        E_min = E_initial
        print('Initial Energy： %s' % E_initial)

        for iter_T in range(annealing_T_number):
            T = exponential_temps[iter_T]
            for iter in range(iteration):
                """calculate the current energy(distance)"""
                E_now = 0
                for j in range(City_nums):
                    if j == (City_nums - 1):
                        middle = Neuron[0, :].reshape((City_nums, 1))
                        E_now += (Neuron[j, :] @ D_matrix @ middle).item()
                    else:
                        middle = Neuron[j + 1, :].reshape((City_nums, 1))
                        E_now += (Neuron[j, :] @ D_matrix @ middle).item()
                """randomly select two cities"""
                city_list = list(range(0, City_nums))
                row_choose = random.sample(city_list, 2)
                i = row_choose[0]
                k = row_choose[1]

                """calculate the energy(distance) after swapping two cities"""
                Neuron[[i, k]] = Neuron[[k, i]]
                E_swapped = 0
                for j in range(City_nums):
                    if j == (City_nums - 1):
                        middle = Neuron[0, :].reshape((City_nums, 1))
                        E_swapped += (Neuron[j, :] @ D_matrix @ middle).item()
                    else:
                        middle = Neuron[j + 1, :].reshape((City_nums, 1))
                        E_swapped += (Neuron[j, :] @ D_matrix @ middle).item()

                if E_swapped < E_now:
                    """Metropolis truncation is performed, which can be omitted in the PSS protocol"""
                    Energy_list.append(E_swapped)  # Store the accepted energy value
                    # Compare with the shortest distance found
                    if E_swapped < E_min:
                        E_min = E_swapped
                        Neuron_min = copy.deepcopy(Neuron)
                else:
                    """Perform 4-node PSS sampling"""
                    co_1 = 0
                    co_2 = 0
                    spin = []
                    for j in range(City_nums):
                        if Neuron[i, j] == 1:
                            co_1 = j
                    for j in range(City_nums):
                        if Neuron[k, j] == 1:
                            co_2 = j
                    """divide the 4 target nodes into two groups"""
                    # The first two nodes of the spin list form one group, and the last two nodes form another group. PSS is established between the two groups.
                    # way 1
                    spin.append(copy.deepcopy([i, co_1]))
                    spin.append(copy.deepcopy([k, co_1]))
                    spin.append(copy.deepcopy([k, co_2]))
                    spin.append(copy.deepcopy([i, co_2]))
                    """
                    # other two grouping ways
                    # way 2
                    spin.append(copy.deepcopy([i, co_1]))
                    spin.append(copy.deepcopy([i, co_2]))
                    spin.append(copy.deepcopy([k, co_1]))
                    spin.append(copy.deepcopy([k, co_2]))
                    # way 3
                    spin.append(copy.deepcopy([i, co_1]))
                    spin.append(copy.deepcopy([k, co_2]))
                    spin.append(copy.deepcopy([i, co_2]))
                    spin.append(copy.deepcopy([k, co_1]))
                    """
                    """Construct the transition matrix between two sets of nodes"""
                    transfer_matrix_0 = np.zeros((4, 4), dtype=float)
                    transfer_matrix_1 = np.zeros((4, 4), dtype=float)
                    spin_1 = spin[0]
                    spin_2 = spin[1]
                    ip = 0
                    for m in range(2):
                        for n in range(2):
                            Neuron[spin[2][0], spin[2][1]] = m
                            Neuron[spin[3][0], spin[3][1]] = n
                            c = self.two_spin_p_stationary_distribution(Neuron, spin_1, spin_2, T)
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
                            Neuron[spin[0][0], spin[0][1]] = m
                            Neuron[spin[1][0], spin[1][1]] = n
                            c = self.two_spin_p_stationary_distribution(Neuron, spin_1, spin_2, T)
                            transfer_matrix_1[0, ip] = (1 - c[0]) * (1 - c[1])
                            transfer_matrix_1[1, ip] = (1 - c[0]) * c[1]
                            transfer_matrix_1[2, ip] = c[0] * (1 - c[1])
                            transfer_matrix_1[3, ip] = c[0] * c[1]
                            ip += 1
                    """Obtain the self-transition matrix for each group of nodes"""
                    transfer_matrix_12 = transfer_matrix_0.dot(transfer_matrix_1)
                    transfer_matrix_34 = transfer_matrix_1.dot(transfer_matrix_0)
                    """Calculate the nonstationary PSS probability distribution (Matrix power multiplication)"""
                    transfer_matrix_12_f = self.matrixPow(transfer_matrix_12, n_transfer)
                    transfer_matrix_34_f = self.matrixPow(transfer_matrix_34, n_transfer)
                    """Calculate the sampling(PSS) probabilities of two valid solutions to the TSP problem"""
                    # entangled way 1
                    column_1 = 2
                    column_2 = 2
                    p1 = transfer_matrix_12_f[1, column_1] * transfer_matrix_34_f[1, column_2]  # Probability of swapping cities
                    p2 = transfer_matrix_12_f[2, column_1] * transfer_matrix_34_f[2, column_2]  # Probability of not swapping cities
                    """
                    # other two grouping ways
                    # way 2
                    column_1 = 2
                    column_2 = 1
                    p1 = transfer_matrix_12_f[1, column_1] * transfer_matrix_34_f[2, column_2]   
                    p2 = transfer_matrix_12_f[2, column_1] * transfer_matrix_34_f[1, column_2]    
                    # way 3
                    column_1 = 3
                    column_2 = 0
                    p1 = transfer_matrix_12_f[0, column_1] * transfer_matrix_34_f[3, column_2]   
                    p2 = transfer_matrix_12_f[3, column_1] * transfer_matrix_34_f[0, column_2] 
                    """

                    p_sample = p1 / (p1 + p2)  # Renormalize PSS probabilities of the two valid solutions
                    if np.random.rand() < p_sample:
                        Neuron[spin[0][0], spin[0][1]] = 0
                        Neuron[spin[1][0], spin[1][1]] = 1
                        Neuron[spin[2][0], spin[2][1]] = 0
                        Neuron[spin[3][0], spin[3][1]] = 1
                    else:
                        Neuron[spin[0][0], spin[0][1]] = 1
                        Neuron[spin[1][0], spin[1][1]] = 0
                        Neuron[spin[2][0], spin[2][1]] = 1
                        Neuron[spin[3][0], spin[3][1]] = 0
                    """
                    # other grouping ways
                    # way 2
                    if np.random.rand() < p_sample:
                        Neuron[spin[0][0], spin[0][1]] = 0
                        Neuron[spin[1][0], spin[1][1]] = 1
                        Neuron[spin[2][0], spin[2][1]] = 1
                        Neuron[spin[3][0], spin[3][1]] = 0
                    else:
                        Neuron[spin[0][0], spin[0][1]] = 1
                        Neuron[spin[1][0], spin[1][1]] = 0
                        Neuron[spin[2][0], spin[2][1]] = 0
                        Neuron[spin[3][0], spin[3][1]] = 1    
                    # way 3
                    if np.random.rand() < p_sample:
                        Neuron[spin[0][0], spin[0][1]] = 0
                        Neuron[spin[1][0], spin[1][1]] = 0
                        Neuron[spin[2][0], spin[2][1]] = 1
                        Neuron[spin[3][0], spin[3][1]] = 1
                    else:
                        Neuron[spin[0][0], spin[0][1]] = 1
                        Neuron[spin[1][0], spin[1][1]] = 1
                        Neuron[spin[2][0], spin[2][1]] = 0
                        Neuron[spin[3][0], spin[3][1]] = 0
                    """

                    """Calculate the system energy after performing the swap or not performing the swap"""
                    E_sampled = 0
                    for j in range(City_nums):
                        if j == (City_nums - 1):
                            middle = Neuron[0, :].reshape((City_nums, 1))
                            E_sampled += (Neuron[j, :] @ D_matrix @ middle).item()
                        else:
                            middle = Neuron[j + 1, :].reshape((City_nums, 1))
                            E_sampled += (Neuron[j, :] @ D_matrix @ middle).item()
                    Energy_list.append(E_sampled)  # Store the accepted energy value
                    """Compare with the shortest distance found"""
                    if E_sampled < E_min:
                        E_min = E_sampled
                        Neuron_min = copy.deepcopy(Neuron)

        print('Shortest distance found: %s' % E_min)
        print('Corresponding path:')
        path_min = []
        for m in range(City_nums):
            for n in range(City_nums):
                if Neuron_min[m, n] == 1.:
                    path_min.append(n)
        print(path_min)
        print('')

        """You can export the list recording the evolution of system energy to an external file for subsequent processing (e.g.: observing the temporal evolution of energy or analyzing the probability distribution of energy)"""
        df = pd.DataFrame({'Iteration': range(1, len(Energy_list) + 1), 'Energy': Energy_list})
        """Append to the CSV file"""
        df.to_csv(f"{self.file_path_output_energy_list}/Nonstationary_PSS_Energy_list_{self.instance_name}.csv", mode='a', index=False, header=True)
        print('The system energy evolution has been exported to an external file')


def main():
    start_time = time.time()
    print('*** Nonstationary_PSS_TSP_solver ***')
    file_path_input_distance_matrix = input("Enter the input file path of distance matrix: ").strip()
    file_path_output_energy_list = input("Enter the output file path of energy evolution list: ").strip()
    print("Available TSP Instances: gr17, gr24, fri26, swiss42, gr48, att48, hk48, berlin52, st70, eil76, rd100, kroA100, kroB100, kroC100, pr107, gr120, ch130, ch150, kroA150, and kroA200")
    instance_name = input("Enter the TSP instance name: ").strip()
    city_nums = int(input("Enter the number of cities: ").strip())
    tsp = TSP(instance_name, city_nums, file_path_input_distance_matrix, file_path_output_energy_list)
    tsp.Nonstationary_PSS_TSP_solver()  # run the main code
    end_time = time.time()
    print('run time：%s' % (end_time - start_time))


if __name__ == "__main__":
    main()
