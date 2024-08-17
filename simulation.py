import numpy as np
import itertools
from copy import deepcopy
import random


class NKModel:
    def __init__(self, N, K, R, G, U, prob_jump=0, Beta=0.1):
        self.N = N  # Number of components
        self.K = K  # Number of interactions
        self.R = R  # Range; Number of different states that constitute the representation

        self.G = G  # Granularity; The smaller the value, the more precise the representation; For example, if the 
                    # granularity value is 2, the representation will be generated based on every 2 states (take their average values)
        
        self.U = U  # Inaccuracy; The standard deviation of the noise added to the predicted value

        self.prob_jump = prob_jump  # The probability of jumping to a random state instead of the best state

        self.Beta = Beta  # The coefficient of cost of changing the state

        self.environment = self.generate_environment()
        self.landscape = self.generate_landscape()
        self.explored = []


    def generate_environment(self):
        """Generate the coefficients for the NK landscape"""
        landscapes = np.random.randn(self.N, 2**(self.K+1))
        return landscapes

    def get_fitness(self, state):
        """Calculate the fitness of a state"""
        total_fitness = 0
        for i in range(self.N):
            # Determine the index for the sub-landscape
            index = state[i]
            for j in range(1, self.K+1):
                interaction_partner = (i + j) % self.N
                index = (index << 1) | state[interaction_partner]
            total_fitness += self.environment[i, index]
        return total_fitness / self.N

    def generate_landscape(self):
        """Generate the actual values of the entire landscape"""
        landscape = {}
        for i in range(2**self.N):
            state = [int(x) for x in f"{i:0{self.N}b}"]
            landscape[tuple(state)] = self.get_fitness(state)
        return landscape

    def generate_changed_series(self, series, num_change):
        """Generate all possible series that can be obtained by changing a specified number of elements in the series"""
        series_list = list(series)
        length = len(series_list)
        index_combinations = itertools.combinations(range(length), num_change)

        changed_series = []
        for indices in index_combinations:
            new_series = series_list[:]
            for index in indices:
                new_series[index] = 1 if new_series[index] == 0 else 0
            if new_series != list(self.current_state):
                changed_series.append(new_series)
        
        return changed_series


    def separate_list_of_lists(self, lst, part_size):
        """Separate a list of lists into n parts"""
        n_parts = (len(lst) + part_size - 1) // part_size
        if n_parts == 0:
            return []
        remainder = len(lst) % n_parts

        parts = []
        start = 0
        for i in range(n_parts):
            end = start + part_size + (1 if i < remainder else 0)
            start = end
            parts.append(lst[start - part_size:end])
        return parts
    

    def calculate_cost(self, state, new_state):
        """Calculate the cost of changing from the current state to the new state"""
        cost = 0
        for i in range(self.N):
            if state[i] != new_state[i]:
                cost += 1
        return cost


    def generate_representation(self, state):
        """
        Generate a representation of the fitness landscape that can be used to predict the performance of areas within the range
        """

        if self.R > 2 ** self.N:
            self.R = 2 ** self.N
        
        representation_lst = []
        representation = {}
        flag = False
        for i in range(0, self.N+1):
            lst = self.generate_changed_series(state, i)
            for j in lst:
                if len(representation_lst) == self.R:
                    flag = True
                    break
                representation_lst.append(j)
            if flag:
                break

        separate_list = self.separate_list_of_lists(representation_lst, self.G)
        for part in separate_list:
            if len(part) == 0:
                break
            average_value = sum([self.landscape[tuple(j)] for j in part]) / len(part)
            for k in part:
                representation[tuple(k)] = average_value

        return representation


    def predict_performance(self, state):
        """
        Based on the representation generated and the inaccuracy, predict the performance of the states within the range
        """
        representation_dic = self.generate_representation(state)
        predicted_dict = deepcopy(representation_dic)
        for key in predicted_dict:
            predicted_dict[key] = predicted_dict[key] * (1 + np.random.normal(0, self.U))
        return predicted_dict
    
    def calculate_nearby_values(self):
        """Calculate the average value of the states within the range"""
        nearby_states = self.generate_changed_series(self.current_state, 1) + self.generate_changed_series(self.current_state, 2)
        average_value = sum([self.landscape[tuple(j)] for j in nearby_states]) / len(nearby_states)
        return average_value
    
    
    def move_agent(self):
        """calculate the best move for the agent based on the predicted performance of the states within the range"""
        neighbors = self.generate_changed_series(self.current_state, 1)
        neighbors_dic = {}
        for neighbor in neighbors:
            neighbors_dic[tuple(neighbor)] = self.landscape[tuple(neighbor)]
        
        best_move = neighbors
        average_value = self.calculate_nearby_values()
        is_local = True
        max_objective = average_value - 1 * self.Beta
        cost = 1 * self.Beta


        # random area with representation
        all_landscapes = set(self.landscape.keys())
        states_set = all_landscapes - set(self.explored)
        state = tuple(random.choice(list(states_set)))
        potential_states = self.predict_performance(state)

        for state_key, state_value in potential_states.items():
            cost_1 = self.Beta * self.calculate_cost(self.current_state, state_key)
            objective = state_value - cost_1
            if objective > max_objective:
                max_objective = objective
                best_move = [state_key]
                cost = cost_1
                is_local = False
            elif objective == max_objective:
                best_move.append(state_key)
                is_local = False
        
        new_state = tuple(random.choice(best_move))
        return new_state, max_objective, cost, neighbors_dic, average_value, is_local


    def simulate_one(self, start=False, print_=False):
        """simulation with print statements"""
        if start:
            initial_state = tuple(np.random.randint(0, 2, self.N))
            self.current_state = initial_state
            self.explored.append(self.current_state)

        new_state, predicted_value, cost, neighbors_dic, average_value, is_local = self.move_agent()
        actual_performance = self.get_fitness(new_state)
        result = actual_performance - cost
        if print_:
            print(f"Current State: {self.current_state}\
                    \nLocal value: {self.landscape[self.current_state]}\
                    \nNew State    : {new_state}\
                    \nCost: {cost}\
                    \nIs Local: {is_local}\
                    \nPredicted Value: {predicted_value}\
                    \nResult: {result}\
                    \nAverage Value: {average_value}\n")
            print("Neighbors:")
            for key, value in neighbors_dic.items():
                print(f"    {key}: {value}")
        self.current_state = new_state
        self.explored.append(self.current_state)

        return result, cost, is_local
    

    def simulate(self, iterations=10):
        """Simulate the agent moving through the landscape"""
        # Random initial state
        initial_state = tuple(np.random.randint(0, 2, self.N))
        self.current_state = initial_state
        self.explored.append(self.current_state)
        results = []
        local = 0

        for _ in range(iterations):
            result, cost, is_local = self.simulate_one()
            results.append([result, cost])
            if is_local:
                local += 1

        return np.mean(results), local / iterations




