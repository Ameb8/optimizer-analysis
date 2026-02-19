import pandas as pd
import numpy as np
from opti_py import Problem

import tomllib
from pathlib import Path


def load_data(config_path: Path) -> pd.DataFrame:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Global
    repetitions = config["global"]["repetitions"]
    init_seed = config["global"]["init_seed"]
    seed_step = config["global"]["seed_step"]
    generations = config["global"]["generations"]
    dimensions = config["global"]["dimensions"]
    problems_cfg = config["global"]["problems"]
    problems = list(problems_cfg.keys())

    # DE
    de_cfg = config["optimizers"]["DE"]
    de_pop = de_cfg["pop_size"]
    de_F = de_cfg["F"]
    de_CR = de_cfg["CR"]
    de_lambda = de_cfg["lambda"]
    mutations = de_cfg["mutations"]
    crossovers = de_cfg["crossovers"]

    de_index = pd.MultiIndex.from_product(
        [
            dimensions,
            problems,
            mutations,
            crossovers,
            range(repetitions),
        ],
        names=["dimension", "problem", "mutation", "crossover", "repetition"],
    )

    df_de = de_index.to_frame(index=False)
    df_de["optimizer"] = "DE"
    df_de["generations"] = generations
    df_de["pop_size"] = de_pop
    df_de["F"] = de_F
    df_de["CR"] = de_CR
    df_de["lambda"] = de_lambda
    df_de["c1"] = np.nan
    df_de["c2"] = np.nan

    # PSO
    pso_cfg = config["optimizers"]["PSO"]
    pso_pop = pso_cfg["pop_size"]
    pso_c1 = pso_cfg["c1"]
    pso_c2 = pso_cfg["c2"]

    pso_index = pd.MultiIndex.from_product(
        [
            dimensions,
            problems,
            range(repetitions),
        ],
        names=["dimension", "problem", "repetition"],
    )

    df_pso = pso_index.to_frame(index=False)
    df_pso["optimizer"] = "PSO"
    df_pso["generations"] = generations
    df_pso["pop_size"] = pso_pop
    df_pso["F"] = np.nan
    df_pso["CR"] = np.nan
    df_pso["lambda"] = np.nan
    df_pso["mutation"] = np.nan
    df_pso["crossover"] = np.nan
    df_pso["c1"] = pso_c1
    df_pso["c2"] = pso_c2

    # Align columns
    common_columns = [
        "optimizer",
        "problem",
        "problem_id",
        "dimension",
        "generations",
        "repetition",
        "seed",
        "lower_bound",
        "upper_bound",
        "pop_size",
        "F",
        "CR",
        "lambda",
        "mutation",
        "crossover",
        "c1",
        "c2",
        "exec_time",
        "best_fitness",
        "fitness_per_iteration",
        "best_solution",
    ]

    df = pd.concat([df_de, df_pso], ignore_index=True)

    # Compute seed
    df["seed"] = init_seed + seed_step * df["repetition"]

    # Add result columns
    df["exec_time"] = np.nan
    df["best_fitness"] = np.nan
    df["fitness_per_iteration"] = None
    df["best_solution"] = None

    # AAdd upper and lower bound columns
    df["lower_bound"] = df["problem"].map(
        lambda p: problems_cfg[p]["lower"]
    )
    df["upper_bound"] = df["problem"].map(
        lambda p: problems_cfg[p]["upper"]
    )

    # Add column for problem ID
    df["problem_id"] = df["problem"].apply(lambda x: int(Problem.from_string(x)))
   
    # Reorder shared columns to start
    df = df[common_columns + [c for c in df.columns if c not in common_columns]]


    return df
