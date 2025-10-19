# from classes import Instance, Chromosome, Population, PopRegister, PivRegister
import os
import sys
import classes
from classes import *

import pandas as pd
from datetime import datetime
import time
import csv


class Simulator:

    # ----- * ----- * ----- * ----- #
    # Initialize Simulator Class
    # ----- * ----- * ----- * ----- #

    def __init__(self,
                 data_name: str,
                 kwargs: dict
                 ):
        
        self.data_name: str = data_name
        self.instance: Instance = Instance(self.data_name)
        
        # Likely to be fixed
        self.pop_size: int = kwargs.get('pop_size')
        self.nb_gen: int = kwargs.get('nb_gen')
        self.mutation_prob: float = kwargs.get('mutation_prob')
        self.elite_ratio: float = kwargs.get('elite_ratio')
        
        # Likely to be tuned
        self.initial_sparsity: float = kwargs.get('initial_sparsity')
        self.nb_pivot: int = kwargs.get('nb_pivot')
        self.pivot_ratio: float = kwargs.get('pivot_ratio')
        self.angle_stdev: float = kwargs.get('angle_stdev')

        # Display and output
        self.display_freq: int = kwargs.get('display_freq')
        self.time_id = datetime.now().strftime("%Y%m%d-%H%M")
        self.result_dir: str = os.path.join(os.getcwd(), f'results/{data_name}_{self.time_id}')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        self.summary_path: str = os.path.join(self.result_dir, 'summary.csv')
        self.stat_path: str = os.path.join(self.result_dir, 'gen_stat.csv')
        self.cr_path: str = os.path.join(self.result_dir, 'chromosome.csv')
        self.pr_path: str = os.path.join(self.result_dir, 'pivot_reg.csv')
        self.edge_idx_path: str = os.path.join(self.result_dir, 'edge_idx.csv')


    # ----- * ----- * ----- #
    # Define Main Methods
    # ----- * ----- * ----- #

    def run(self):
        
        self.nb_sample_per_pivot: int = int(self.pop_size * 
                                            (1-self.elite_ratio) / self.nb_pivot)
        self.nb_elite_chromosomes: int = int(self.pop_size * self.elite_ratio)
        self.instance.display_info()
        
        self.stat_result: list = list()
        self.cr_result: list = list()
        self.pr_result: list = list()

        start_time, gen_idx = time.time(), 0
        pop = Population(instance=self.instance,
                         population_size=self.pop_size,
                         initial_sparsity=self.initial_sparsity,)
        pop.add_initial_chromosomes(children_size=self.pop_size)
        pop.sort_chromosomes()
        best_found_chr: Chromosome = pop.chromosomes[0]

        pop.calculte_statistics(gen_idx=gen_idx)
        pop.save_chromosome_info(gen_idx=gen_idx)

        while gen_idx < self.nb_gen:
            
            pop.select_pivots(nb_pivot=self.nb_pivot,
                              max_percentage=self.pivot_ratio)
            pop_reg = PopRegister(population=pop)
            if gen_idx % self.display_freq == 0 or gen_idx == self.nb_gen - 1:
                print(f"\n>>> Generation {gen_idx:03d}")
                pop.display_stats()  
            
            self.stat_result.append(pop.stats)
            self.cr_result += pop.chromosome_info 
            
            gen_idx += 1
            new_pop = Population(instance=self.instance,
                                 population_size=self.pop_size)
            new_pop.chromosomes = [c for c in 
                                   pop.chromosomes[:self.nb_elite_chromosomes]]
            for c in new_pop.chromosomes:
                c.piv_reg_idx = 0
            for i, pivot in enumerate(pop.pivots, start=1):
                pivot_reg = PivRegister(pop_reg=pop_reg,
                                        pivot_idx=pivot[0],
                                        pivot_chromosome=pivot[1],
                                        angle_stdev=self.angle_stdev,
                                        mutation_prob=self.mutation_prob)
                pivot_reg.save_info(gen_idx=gen_idx, piv_idx=i)
                self.pr_result.append(pivot_reg.info)
                
                new_pop.add_new_chromosomes(children_size=self.nb_sample_per_pivot,
                                            pivot_prob_arr=pivot_reg.pivot_prob_arr,
                                            piv_reg_idx=i)
            new_pop.sort_chromosomes()
            if new_pop.chromosomes[0].fitness > best_found_chr.fitness:
                best_found_chr = new_pop.chromosomes[0]
            
            new_pop.calculte_statistics(gen_idx=gen_idx)
            new_pop.save_chromosome_info(gen_idx=gen_idx)
              
            
            pop = new_pop
        
        end_time = time.time()
        
        sim_result = dict()
        sim_result['timestamp'] = self.time_id
        sim_result['data_name'] = self.data_name
        sim_result['best_fitness'] = best_found_chr.fitness
        sim_result['best_cost'] = best_found_chr.cost
        sim_result['optimal_cost'] = self.instance.optimal_cost
        sim_result['best_optimality_gap'] = best_found_chr.optimality_gap
        sim_result['best_edge_density'] = best_found_chr.edge_density
        sim_result['best_node_density'] = best_found_chr.node_density
        sim_result['best_sol'] = ''.join(best_found_chr.sol.astype(str))
        sim_result['total_time_sec'] = end_time - start_time

        with open(self.summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for key, value in sim_result.items():
                    writer.writerow([key, value])

    def dump_results(self):
        pd.DataFrame(self.stat_result).to_csv(self.stat_path, index=False, float_format="%.2f")
        pd.DataFrame(self.cr_result).to_csv(self.cr_path, index=False, float_format="%.2f")
        pd.DataFrame(self.pr_result).to_csv(self.pr_path, index=False, float_format="%.2f")
        pd.DataFrame([(k, f"{u},{v}") 
                      for k, (u, v) in self.instance.idx_edge_dict.items()], 
                      columns=["idx", "edge"]).to_csv(self.edge_idx_path, index=False, float_format="%.2f")


if __name__ == "__main__":

    config = {
        'initial_sparsity': (0.9, 0.8, 0.7),  # param tune
        'pop_size': 500,
        'nb_gen': 500,
        'mutation_prob': 0.01,
        'elite_ratio': 0.05,
        'nb_pivot': 4,  # param tune
        'pivot_ratio': 0.05,  # param tune
        'angle_stdev': 0.05,  # param tune
        'display_freq': 1
    }

    simulator = Simulator(data_name='se03',
                          kwargs=config)
    simulator.run()
    simulator.dump_results()
