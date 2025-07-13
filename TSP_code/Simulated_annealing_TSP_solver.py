"""
Simulated annealing for TSP
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
        self.T0 = None
        self.alpha = None
        self.d_matrix = None
        self.load_instance()
        self.load_parameters()

    def load_parameters(self):
        """Load parameter settings based on the instance name, including the initial temperature for exponential annealing T0, and the exponential factor alpha"""
        parameter_mapping = {
            "gr17": {"T0": 100, "alpha": 0.976}, "gr24": {"T0": 100, "alpha": 0.955},
            "fri26": {"T0": 80, "alpha": 0.968}, "swiss42": {"T0": 50, "alpha": 0.968},
            "gr48": {"T0": 200, "alpha": 0.972}, "att48": {"T0": 1000, "alpha": 0.963},
            "hk48": {"T0": 1000, "alpha": 0.968}, "berlin52": {"T0": 800, "alpha": 0.966},
            "st70": {"T0": 50, "alpha": 0.955}, "eil76": {"T0": 50, "alpha": 0.956},
            "rd100": {"T0": 300, "alpha": 0.964}, "kroA100": {"T0": 2000, "alpha": 0.955},
            "kroB100": {"T0": 2000, "alpha": 0.955}, "kroC100": {"T0": 2000, "alpha": 0.955},
            "pr107": {"T0": 4000, "alpha": 0.958}, "gr120": {"T0": 300, "alpha": 0.964},
            "ch130": {"T0": 200, "alpha": 0.972}, "ch150": {"T0": 200, "alpha": 0.972},
            "kroA150": {"T0": 2000, "alpha": 0.955}, "kroA200": {"T0": 1000, "alpha": 0.962}
            # You can change the running parameters or add other instances
        }

        if self.instance_name in parameter_mapping:
            params = parameter_mapping[self.instance_name]
            self.T0 = params["T0"]
            self.alpha = params["alpha"]
        else:
            raise ValueError(f"No parameter settings available for instance: {self.instance_name}")
        print(
            f"Parameters loaded for {self.instance_name}: initial_temperature: T0={self.T0}, exponential_annealing_factor: alpha={self.alpha} (adjustable)")

    def load_instance(self):
        """Load the specified TSP instance and compute the corresponding distance matrix"""
        if self.instance_name == "swiss42":
            file_path = f"{self.file_path_input_distance_matrix}/{self.instance_name}-matrix-DIY.txt"
            self.d_matrix = np.array(self.read_distance_type_MATRIX(file_path)).reshape(
                (self.city_nums, self.city_nums))
        elif self.instance_name in ["berlin52", "st70", "eil76", "rd100", "kroA100", "kroB100", "kroC100", "pr107",
                                    "ch130", "ch150", "kroA150", "kroA200"]:
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

    def SA_run(self):
        City_nums = self.city_nums
        D_matrix = self.d_matrix
        T0 = self.T0
        alpha = self.alpha
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
                    """Perform Metropolis truncation"""
                    Energy_list.append(E_swapped)  # Store the accepted energy value
                    # Compare with the shortest distance found
                    if E_swapped < E_min:
                        E_min = E_swapped
                        Neuron_min = copy.deepcopy(Neuron)
                else:
                    """Perform stochastic sampling"""
                    p_sample = 1 / np.exp((E_swapped - E_now) / T)
                    if np.random.rand() > p_sample:
                        Neuron[[i, k]] = Neuron[[k, i]]

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
        df.to_csv(f"{self.file_path_output_energy_list}/SA_Energy_list_{self.instance_name}.csv", mode='a', index=False, header=True)
        print('The system energy evolution has been exported to an external file')


def main():
    start_time = time.time()
    print('*** SA_TSP ***')
    file_path_input_distance_matrix = input("Enter the input file path of distance matrix: ").strip()
    file_path_output_energy_list = input("Enter the output file path of energy evolution list: ").strip()
    print("Available TSP Instances: gr17, gr24, fri26, swiss42, gr48, att48, hk48, berlin52, st70, eil76, rd100, kroA100, kroB100, kroC100, pr107, gr120, ch130, ch150, kroA150, and kroA200")
    instance_name = input("Enter the TSP instance name: ").strip()
    city_nums = int(input("Enter the number of cities: ").strip())
    tsp = TSP(instance_name, city_nums, file_path_input_distance_matrix, file_path_output_energy_list)
    tsp.SA_run()  # run the main code
    end_time = time.time()
    print('run time：%s' % (end_time - start_time))


if __name__ == "__main__":
    main()

