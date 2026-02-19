from opti_py import DifferentialEvolution, ParticleSwarm, Problem, ExperimentConfig
import pandas as pd
import numpy as nd

from pathlib import Path
import time

from typing import NamedTuple

from .load_experiments import load_data


def init_config(exp: NamedTuple) -> ExperimentConfig:
    return ExperimentConfig(
        problem_type=exp["problem_id"],
        dimensions=exp["dimension"],
        lower=exp["lower_bound"],
        upper=exp["upper_bound"],
        max_iterations=exp["repetition"],
        seed=exp["seed"]
    )


def init_de(exp: NamedTuple) -> DifferentialEvolution:
    return opti_py.DifferentialEvolution(
        init_config(exp),
        scale=exp["F"],
        crossover_rate=exp["CR"],
        pop_size=exp["pop_size"],
        mutation=exp["mutations"],
        crossover=exp["crossovers"]
    )


def init_pso(exp: NamedTuple) -> ParticleSwarm:
    return opti_py.ParticleSwarm(
        init_config(exp),
        c1=exp["c1"],
        c2=exp["c2"],
        pop_size=exp["pop_size"]
    )

def run_experiment(exp: NamedTuple) -> bool:
    exp_config: ExperimentConfig = ExperimentConfig(
        problem_type=exp["problem_id"],
        dimensions=exp["dimension"],
        lower=exp["lower_bound"],
        upper=exp["upper_bound"],
        max_iterations=exp["repetition"],
        seed=exp["seed"]
    )

    # Run experiment
    if exp["optimizer"] == "PSO":
        optimizer = init_pso(exp)
    elif exp["optimizer"] == "DE":
        optimizer = init_de(exp)
    else
        return False
    
    # Run timed optimization
    start: float = time.perf_counter()
    fitnesses = optimizer.optimize()
    end: float = time.perf_counter()

    # Write experiment results
    exp["exec_time"] = end - start
    exp["best_fitness"] = optimizer.get_best_fitness()
    exp["fitness_per_iteration"] = fitnesses
    exp["best_solution"] = optimizer.get_best_solution()
    
    return True
    
    

def run_experiments(config_path: Path) -> bool:
    experiments: pd.DataFrame = load_data(config_path)

    # Iterate through rows
    for exp in experiments.itertuples(index=True):
        if not run_experiment(exp):
            return False # Error running experiment
        

    return True

'''
import opti_py

config = opti_py.ExperimentConfig(
    problem_type=1,
    dimensions=30,
    lower=-100,
    upper=100,
    max_iterations=5000,
    seed=42
)

de = opti_py.DifferentialEvolution(
    config,
    pop_size=200,
    scale=0.7,
    crossover_rate=0.6
)

fitnesses = de.optimize()

print(f"\nFitness Values:\n\n\n{fitnesses}")

best_fitness = de.get_best_fitness()
best_solution = de.get_best_solution()

print("Best fitness:", best_fitness)
print("Best solution:", best_solution)
'''


'''
df.loc[i, "exec_time"] = elapsed
df.loc[i, "best_fitness"] = best
df.loc[i, "fitness_per_iteration"] = fitness_curve  # numpy array
df.loc[i, "best_solution"] = best_vector
'''