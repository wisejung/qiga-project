# %%
# Import Libraries
import os
import statistics

import random as rd
import networkx as nx
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

from gurobipy import GRB
from numpy.typing import NDArray
from math import *


# %%
class Instance:

    def __init__(self, 
                 data_name: str
                 ):
          
        # ----- * ----- * ----- * ----- #
        # Read Input Files (⚠️ HARDCODED!)
        # ----- * ----- * ----- * ----- #
        
        self.data_name = data_name
        print(os.getcwd())
        self.file_path = os.path.join(os.getcwd(),
                                      f"input/stp-sp/{self.data_name}.stp")
        self.opt_dir = os.path.join(os.getcwd(), "input/stp-sp_grb-results")
        
        
        # ----- * ----- * ----- * ----- #
        # Initialize Class Variables
        # ----- * ----- * ----- * ----- #
        
        # Networkx objects
        self.graph: nx.Graph = nx.Graph()
        self.terminals: list[int] = list()
        self.idx_node_dict: dict[int:int] = dict()
        self.idx_edge_dict = dict()
        self.pos = None

        # Gurobi objects
        self._grb_model: gp.Model = gp.Model(self.data_name)
        self._grb_var: dict = dict()
        self._grb_optimum: int = 0

        # Instance info
        # self.timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        self.sum_cost = None
        self.nb_node = None
        self.nb_edge = None
        self.nb_terminal = None

        # Optimal solution info
        self.optimal_cost = None
        self.optimal_arc_num = None
        self.optimal_fitness = None
        self.optimal_nb_edge = None

        # Run internal methods
        self._parse_stp_to_dict()
        self._get_idx_edge_dict()
        self._get_cost_total_sum()

        # Solve GRB model or read optimal solution
        solution_path = os.path.join(self.opt_dir, f"{self.data_name}_sol.txt")
        if not os.path.isfile(solution_path):
            self._optimize()
            self._get_optimum()
        else:
            self._get_optimum()


    # ----- * ----- * ----- #
    # Define Internal Methods
    # ----- * ----- * ----- #

    # Read input files and save data into class variables and networkx objects
    def _parse_and_save_old(self):
        data = list()
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                numbers = list(map(int, line.split()))
                data.append(tuple(numbers))

        line_idx = 0
        while line_idx < len(data):
            if line_idx == 0:
                self.nb_node, self.nb_edge = data[line_idx]
                self.graph.add_nodes_from(range(1, 1+self.nb_node))
            elif 1 <= line_idx < self.nb_edge + 1:
                from_node, to_node, weight = data[line_idx]
                self.graph.add_edge(from_node, to_node, weight=weight)
            elif line_idx == self.nb_edge + 1:
                self.nb_terminal = data[line_idx]
            elif line_idx > self.nb_edge + 1:
                self.terminals.extend(data[line_idx])
            else:
                break
            line_idx += 1

        for node in self.graph.nodes():
            if node in self.terminals:
                self.graph.nodes[node]['terminal'] = True
            else:
                self.graph.nodes[node]['terminal'] = False
        self.pos = nx.circular_layout(self.graph)
    
    def _parse_stp_to_dict(self) -> dict:
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        
        section = None
        for ln in lines:
            if ln.upper() == "EOF":
                break
            if ln.upper() == "END":
                section = None
                continue
            if ln.upper().startswith("SECTION"):
                section = ln.split()[1].lower()
                continue

            if section == "comment":
                pass
                """
                parts = ln.split(maxsplit=1)
                if len(parts) == 2:
                    key, val = parts
                    data["comment"][key.lower()] = val.strip('"')
                """

            elif section == "graph":
                parts = ln.split()
                if parts != []:
                    if parts[0].lower() == "nodes":
                        self.nb_node = int(parts[1])
                    elif parts[0].lower() == "edges":
                        self.nb_edge = int(parts[1])
                    elif parts[0].upper() == "E": 
                        u, v, w = int(parts[1]), int(parts[2]), float(parts[3])
                        self.graph.add_edge(u, v, weight=w)

            elif section == "terminals":
                parts = ln.split()
                if parts[0].upper() == "TERMINALS":
                    self.nb_terminal = int(parts[1])
                elif parts[0].upper() == "T":
                    self.terminals.append(int(parts[1]))
            
        for node in self.graph.nodes():
            if node in self.terminals:
                self.graph.nodes[node]['terminal'] = True
            else:
                self.graph.nodes[node]['terminal'] = False
        self.pos = nx.circular_layout(self.graph)


    # Create an edge index dictionary for easy access
    def _get_idx_edge_dict(self):
        edges = list(self.graph.edges())  # freeze order
        self.idx_edge_dict = dict(enumerate(edges))
    
    # Get maximum costs among edges to calculate fitness
    def _get_cost_total_sum(self):
        self.sum_cost = sum([self.graph.get_edge_data(e[0], e[1])['weight'] for e in self.graph.edges])

    # If solution file already exists
    # -> Get optimum from the solution file
    def _get_optimum(self):
        sol_path = os.path.join(self.opt_dir, f"{self.data_name}_sol.txt")
        with open(sol_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        header = next((ln for ln in lines if ln.upper().startswith("OBJ")), None)
        if header is None:
            raise ValueError("Solution file missing 'OBJ' header line.")

        self.optimal_cost = int(float(header.split()[1]))
        edge_lines = [ln for ln in lines if not ln.upper().startswith("OBJ")]
        self.optimal_nb_edge = len(edge_lines)
        self.optimal_fitness = self.sum_cost - self.optimal_cost

    # If solution file not exists yet
    # -> Solve the GRB model and save the solution
    def _optimize(self):
        # Set logger parmameters
        log_path = os.path.join(self.opt_dir, f"{self.data_name}_log.txt")
        sol_path = os.path.join(self.opt_dir, f"{self.data_name}_sol.txt")
        self._grb_model.setParam("LogFile", log_path)
        
        # Define the model
        for edge in self.graph.edges:
            e = (edge[0], edge[1])
            self._grb_var[e] = self._grb_model.addVar(vtype=GRB.BINARY, name=f"x_{e[0]},{e[1]}")
        objective = gp.quicksum([self.graph.edges[e]['weight'] * self._grb_var[e]
                                 for e in self.graph.edges])
        self._grb_model.setObjective(objective, GRB.MINIMIZE)
        for t in self.terminals:
            tmp_lhs, var_keys = 0, list(self._grb_var.keys())
            for edge in self.graph.edges(t, data=True):
                e, e_flipped = (edge[0], edge[1]), (edge[1], edge[0])
                if e in var_keys:
                    tmp_lhs += self._grb_var[e]
                else:
                    tmp_lhs += self._grb_var[e_flipped]
            self._grb_model.addConstr(tmp_lhs >= 1)
        self._grb_model.optimize()        
        self._grb_optimum = self._grb_model.ObjVal
        
        # Save the obtained solution
        with open(sol_path, "w", encoding="utf-8") as file:
            file.write(f"OBJ {self._grb_optimum}\n")
            for key in list(self._grb_var.keys()):
                if self._grb_var[key].x > 0.5:
                    file.write(f"{key[0]} {key[1]}\n")        


    # ----- * ----- * ----- #
    # Define Display Methods
    # ----- * ----- * ----- #

    # Display instance information if needed
    def display_info(self):
        print("\n>>> INSTANCE INFO")
        print(f"1. Name: {self.data_name}")
        print(f"2. Nodes: {len(self.graph.nodes)}")
        print(f"3. Edges: {len(self.graph.edges)}")
        print(f"4. Terminal Nodes: {len(self.terminals)}")

    """
    # Display instance graph and save if needed
    def display_and_save_graph(self,
                               display_flag: bool = True,
                               save_flag: bool = False):
        
        node_colors = ['red' if self.graph.nodes[node]['terminal'] 
                       else 'grey' 
                       for node in self.graph.nodes()]
        # edge_labels = nx.get_edge_attributes(instance.graph, 'weight')

        nx.draw(self.graph, 
                self.pos, 
                # with_labels=True, 
                node_color=node_colors, 
                node_size=20, 
                font_size=2, 
                # font_weight='bold'
                )

        if save_flag:
            results_dir = os.path.join(os.getcwd(), 'results')
            file_name = f'{self.timestamp}_instance_graph_{self.data_name}.png'
            plt.savefig(os.path.join(results_dir, file_name), 
                        dpi=300, 
                        bbox_inches="tight")
        if display_flag:
            plt.show()
        
        plt.close()
    """

# %%
class Chromosome:

    def __init__(self, 
                 instance: Instance,
                 sample: np.ndarray, 
                 piv_reg_idx: int | None  # Parent pivot chromosome idx
                 ):

        # Initialize solution info
        self.sol: np.ndarray = sample
        self.piv_reg_idx: int = piv_reg_idx
        self.instance: Instance = instance
        self.nb_var: int = self.instance.nb_edge
        self.fitness = None
        self.cost = None
        self.feasibility = None

        # Initialize networkx objects
        self.graph = None
        self.pos = None
        
        # Run internal methods
        self._get_graph()
        self._compute_fitness()

        self.edge_density: float = sum(self.sol) / self.instance.nb_edge
        self.node_density: float = 1 - len(list(nx.isolates(self.graph))) / self.instance.nb_node


    # ----- * ----- * ----- #
    # Define Internal Methods
    # ----- * ----- * ----- #

    def _is_feasible(self) -> bool:

        used_nodes = [n for n, d in self.graph.degree() if d > 0]
        H = self.graph.subgraph(used_nodes).copy()
        self._feas_subgraph = H

        if H.number_of_edges() == 0:
            self.num_components = 0
            self.feasibility_tree = False
            self.feasibility_terminal = False
            return False

        comps = list(nx.connected_components(H))
        self.num_components = len(comps)
        if self.num_components != 1:
            self.feasibility_tree = False
            self.feasibility_terminal = all(t in H for t in self.instance.terminals)
            return False

        self.feasibility_tree = nx.is_tree(H)
        self.feasibility_terminal = all(t in H for t in self.instance.terminals)

        return self.feasibility_tree and self.feasibility_terminal


    # Create graph based on the solution array
    def _get_graph(self):
        
        self.graph = nx.Graph()
        self.pos = self.instance.pos  # Use same pos setting as base instance
        self.graph.add_nodes_from(self.instance.graph.nodes)
        
        for i, value in enumerate(self.sol):
            if value == 1:
                u_node, v_node = self.instance.idx_edge_dict[i]
                weight_val = self.instance.graph.get_edge_data(u_node, v_node)['weight']
                self.graph.add_edge(u_node, v_node, weight=weight_val)

        for node in self.graph.nodes():
            if node in self.instance.terminals:
                self.graph.nodes[node]['terminal'] = True
            else:
                self.graph.nodes[node]['terminal'] = False

    # Compute fitness based on the given instance
    def _compute_fitness(self):
        if self._is_feasible():
            self.cost = sum(data['weight'] 
                            for _, _, data in self.graph.edges(data=True))
            self.fitness = self.instance.sum_cost - self.cost
            self.feasibility = True
            self.optimality_gap = self.cost / self.instance.optimal_cost - 1
        else:
            self.cost = 0
            self.fitness = 0
            self.feasibility = False
            self.optimality_gap = None
        

    # ----- * ----- * ----- #
    # Define Display Methods
    # ----- * ----- * ----- #
    
    # Display chromosome information if needed
    def display_info(self):
        print("\n>>> CHROMOSOME INFO")
        print(f"1. Feasibility: {self.feasibility}")
        print(f"2. Fitness: {self.fitness}")
        print(f"3. Cost: {self.cost}")
        print(f"4. Edge Density: {self.edge_density*100:.2f}%")
        print(f"5. Node Density: {self.node_density*100:.2f}%")
    



# %%
class Population:

    def __init__(self, 
                 instance: Instance,
                 population_size: int,
                 initial_sparsity: tuple[float] = None
                 ):
        
        # Initialize population information
        self.instance = instance
        self.population_size = population_size
        self.chromosomes = list()
        self.pivots = list()
        
        self.is_sorted = False
        if initial_sparsity is not None:
            self.initial_sparsity = initial_sparsity


    # ----- * ----- * ----- #
    # Define Main Methods
    # ----- * ----- * ----- #
    
    def add_initial_chromosomes(self,
                                children_size):
        probs = list()
        for idx in range(self.instance.nb_edge):
            edge = self.instance.idx_edge_dict[idx]
            terminals = self.instance.terminals
            if edge[0] in terminals and edge[1] in terminals:
                probs.append(self.initial_sparsity[0]) # Highest chance
            elif edge[0] not in terminals and edge[1] not in terminals:
                probs.append(self.initial_sparsity[2]) # Lowest chance
            else:
                probs.append(self.initial_sparsity[1])
        
        for _ in range(children_size):
            new_sample = (np.random.rand(len(probs)) < probs).astype(int)
            new_chromosome = Chromosome(instance=self.instance,
                                        sample=new_sample,
                                        piv_reg_idx=None)
            self.chromosomes.append(new_chromosome)

    def add_new_chromosomes(self,
                            children_size: int,
                            pivot_prob_arr,  # Get from PivotRegister.pivot_prob_arr
                            piv_reg_idx):  # Get from PivotRegister.piv_reg_idx
        for _ in range(children_size):
            new_sample = list()
            for gene_prob_arr in pivot_prob_arr:
                if rd.random() < gene_prob_arr[0]:
                    new_sample.append(0)
                else:
                    new_sample.append(1)
            tmp_child = Chromosome(instance=self.instance,
                                    sample=np.array(new_sample),
                                    piv_reg_idx=piv_reg_idx)
            self.chromosomes.append(tmp_child)
    
    def sort_chromosomes(self):
        self.chromosomes = sorted(self.chromosomes, 
                                  key=lambda x: x.fitness, 
                                  reverse=True)
        
        infeasible = [c for c in self.chromosomes if not c.feasibility]
        feasible = [c for c in self.chromosomes if c.feasibility]
        infeasible = sorted(infeasible, 
                            key=lambda c: c.nb_edge,
                            reverse=False)

        self.is_sorted = True
        self.population_size = len(self.chromosomes)

    def select_pivots(self,
                      nb_pivot: int,
                      max_percentage: float):
        
        assert self.is_sorted, "Population not sorted"  # Check if chromosomes are sorted
        nb_candidates = int(len(self.chromosomes) * max_percentage)
        candidates = [(i, c) for i, c in enumerate(self.chromosomes[:nb_candidates]) if c.fitness > 0]
        assert len(candidates) > 0, "No feasible chromosome within range"  # Check if chromosomes are sorted
        
        if len(candidates) < nb_pivot:
            extra = rd.choices(candidates, k=nb_pivot - len(candidates))
            self.pivots = candidates + extra
        else:
            self.pivots = rd.sample(candidates, k=nb_pivot) # element: ([idx_from_pop, chromosome_object])

    
    def calculte_statistics(self, gen_idx: int):
        
        # Calculate necessary statistics
        best_fitness = max([c.fitness for c in self.chromosomes])
        best_cost = min([c.cost for c in self.chromosomes if c.cost > 0])
        best_optgap = (best_cost - self.instance.optimal_cost)/self.instance.optimal_cost
        avg_fitness = statistics.mean([c.fitness for c in self.chromosomes])
        avg_cost = statistics.mean([c.cost for c in self.chromosomes if c.cost != None])
        avg_nb_edge = statistics.mean([len(c.graph.edges) for c in self.chromosomes])
        feasible_solution_ratio = len([c for c in self.chromosomes if c.feasibility]) / len(self.chromosomes)

        current_statistics = {'gen_idx': gen_idx,
                              'best_fitness': best_fitness,
                              'best_cost': best_cost,
                              'optimality_gap': best_optgap,
                              'avg_fitness': avg_fitness,
                              'avg_cost': avg_cost,
                              'avg_nb_edge': avg_nb_edge,
                              'feasibility_ratio': feasible_solution_ratio}
        self.stats = current_statistics
    
    def display_stats(self):
        for (key, value) in self.stats.items():
            if key == 'gen_idx':
                continue
            elif key in ['optimality_gap', 'feasibility_ratio']:
                print(f"* {key}: {value:.4f}")
            else:
                print(f"* {key}: {value:.2f}")
    
    def save_chromosome_info(self, gen_idx: int):
        self.chromosome_info: list[dict] = list()
        for i, chromosome in enumerate(self.chromosomes):
            c_info = dict()
            c_info['gen'] = gen_idx
            c_info['idx'] = i
            c_info['fitness'] = chromosome.fitness
            c_info['cost'] = chromosome.cost
            c_info['is_tree'] = chromosome.feasibility_tree
            c_info['includes_terminals'] = chromosome.feasibility_terminal
            c_info['feasibility'] = chromosome.feasibility
            c_info['pivreg_idx'] = chromosome.piv_reg_idx
            c_info['node_density'] = chromosome.node_density
            c_info['edge_density'] = chromosome.edge_density
            c_info['sol'] = ''.join(chromosome.sol.astype(str))
            self.chromosome_info.append(c_info)
        

#%%
class PopRegister:

    def __init__(self,
                 population: Population):
        
        self.population = population
        self.chromosomes = self.population.chromosomes
        assert len(self.chromosomes) > 0, "Population is empty"

        self.pop_size: int = len(self.chromosomes)
        self.nb_var: int = len(self.chromosomes[0].sol)
        self.prob_arr: NDArray[np.float64] = self._get_prob_arr()
        self.coeff_arr: NDArray[np.float64] = self._get_coeff_arr()


    # ----- * ----- * ----- #
    # Define Internal Methods
    # ----- * ----- * ----- #

    def _get_prob_arr(self):
        subpopulation_arr = np.vstack([chromosome.sol 
                                       for chromosome in self.population.chromosomes])
        prob_one_arr = np.sum(np.array(subpopulation_arr), axis=0) / self.pop_size
        return np.column_stack((1-prob_one_arr, prob_one_arr))

    def _get_coeff_arr(self):
        return np.sqrt(self.prob_arr)


class PivRegister:

    def __init__(self,
                 pop_reg: PopRegister,
                 pivot_idx: int, # Index from prev population
                 pivot_chromosome: tuple,  # Pivot chromosome object
                 angle_stdev: float,  # Parameter
                 mutation_prob: float  # Parameter
                 ):
        
        self.pop_reg = pop_reg
        self.prob_arr = pop_reg.prob_arr
        self.coeff_arr = pop_reg.coeff_arr

        self.pivot_idx = pivot_idx
        self.pivot_arr: np.ndarray = pivot_chromosome.sol
        self.pivot_coeff_arr = self._get_pivot_register(angle_stdev, mutation_prob)
        self.pivot_prob_arr = self.pivot_coeff_arr ** 2


    # ----- * ----- * ----- #
    # Define Internal Methods
    # ----- * ----- * ----- #

    def _apply_pivot_to_gene(self,
                             gene_arr: np.ndarray,
                             gene_pivot_val: int,
                             rotation_stdev: float,
                             mutation_prob: float):
        tmp_unitary: np.ndarray = self._unitary(gene_arr,
                                                gene_pivot_val,
                                                rotation_stdev,
                                                mutation_prob)
        return np.dot(tmp_unitary, gene_arr)

    def _get_pivot_register(self,
                            angle_stdev: float,
                            mutation_prob: float):
        new_pivot_register = list()
        for gene_pivot_val, gene_coeff_arr in zip(self.pivot_arr, self.coeff_arr):
            new_gene_arr = self._apply_pivot_to_gene(gene_coeff_arr,
                                                     gene_pivot_val,
                                                     angle_stdev,
                                                     mutation_prob)
            new_pivot_register.append(new_gene_arr)
        return np.array(new_pivot_register)
    
    def save_info(self,
                  gen_idx: int,
                  piv_idx: int):
        self.info = dict()
        self.info['gen_idx'] = gen_idx
        self.info['reg_idx'] = piv_idx
        self.info['pivot_idx'] = self.pivot_idx
        for i, arr in enumerate(self.pivot_coeff_arr):
            self.info[f'alpha_{i}'] = str(float(arr[0]))
            self.info[f'beta_{i}'] = str(float(arr[1]))


    # ----- * ----- * ----- #
    # Unitary Operations
    # ----- * ----- * ----- #

    @staticmethod
    def _plus_rotation_gate(angle_stdev):
        theta = abs(np.random.normal(loc=0.0, 
                                     scale=angle_stdev * pi, size=1)[0])
        rotation_gate = np.array([[cos(theta), -sin(theta)], 
                                  [sin(theta), cos(theta)]])
        return rotation_gate

    @staticmethod
    def _minus_rotation_gate(angle_stdev):
        theta = (-1) * abs(np.random.normal(loc=0.0, 
                                            scale=angle_stdev * pi, size=1)[0])
        rotation_gate = np.array([[cos(theta), -sin(theta)], 
                                  [sin(theta), cos(theta)]])
        return rotation_gate

    @staticmethod
    def _reflection_gate(gene_arr: np.ndarray):
        alpha = gene_arr[0]
        theta: float = 2 * (0.25 * pi - acos(alpha))
        rotation_gate = np.array([[cos(theta), -sin(theta)], 
                                  [sin(theta), cos(theta)]])
        return rotation_gate

    def _unitary(self,
                 gene_arr: np.ndarray,
                 gene_pivot: int,
                 angle_stdev: float,
                 mutation_prob: float):
        alpha = gene_arr[0]
        if rd.random() > mutation_prob:  # No mutation
            if gene_pivot == 0 and alpha >= 1 / sqrt(2):
                # Same side / Lower area / Perturb only
                return self._plus_rotation_gate(angle_stdev)
            elif gene_pivot == 0 and alpha < 1 / sqrt(2):
                # Opposite side / Upper area / Reflect and Perturb
                return self._reflection_gate(gene_arr)
            elif gene_pivot == 1 and alpha > 1 / sqrt(2):
                # Opposite side / Lower area / Reflect and Perturb
                return self._reflection_gate(gene_arr)
            else:
                # Same upper side / Upper area / Perturb only
                return self._plus_rotation_gate(angle_stdev)
        else:  # Mutation
            if gene_pivot == 0 and alpha > 1 / sqrt(2):
                # Same side / Lower area / Reflect and Perturb
                return self._reflection_gate(gene_arr)
            elif gene_pivot == 0 and alpha <= 1 / sqrt(2):
                # Opposite side / Upper area / Perturb only
                return self._plus_rotation_gate(angle_stdev)
            elif gene_pivot == 1 and alpha >= 1 / sqrt(2):
                # Opposite side / Lower area / Perturb only
                return self._minus_rotation_gate(angle_stdev)
            else:
                # Same upper side / Upper area / Reflect and Perturb
                return self._reflection_gate(gene_arr)


# ----- * ----- * ----- * ----- #
# Class Testing Snippet
# ----- * ----- * ----- * ----- #
if __name__ == "__main__":

    for e_num in [100, 500]:

        break
        
        # Read Instance
        #instance = Instance(data_group='b', data_idx=1, augmented_edge=e_num)
        instance = Instance(data_name="design432")

        # Display Instance Info
        instance.display_info()
        instance.display_and_save_graph(display_flag=False,
                                        save_flag=True)