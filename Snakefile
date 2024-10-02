from os.path import join as j
import numpy as np
import itertools
import pandas as pd
from snakemake.utils import Paramspace

configfile: "workflow/config.yaml"

# Import utilities
include: "workflow/workflow_utils.smk"

DATA_DIR = config["data_dir"]
FIG_DIR = config["fig_dir"]

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Parameters for the consensus dynamics simulations (SBM)
params_matrix_weighted_sbm = {
    "n_nodes": [120],
    "dim": [2, 3,  10],
    "pin": [0.3],
    "pout": [0.3],
    "noise": [0.1],
    "coherence": [1, 0.8],
    "n_communities": [1, 2, 3]
}
paramspace_matrix_weighted_sbm = to_paramspace(params_matrix_weighted_sbm)

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
# Results for the consensus dynamics simulations (SBM)
RES_CONS_DYN = j(DATA_DIR, "consensus-dynamics", f"sbm-{paramspace_matrix_weighted_sbm.wildcard_pattern}.npz")
RES_RANDOM_WALK = j(DATA_DIR, "random-walk", f"sbm-{paramspace_matrix_weighted_sbm.wildcard_pattern}.npz")

# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
# Figures for the consensus dynamics simulations (SBM)
FIG_CONS_DYN = j(FIG_DIR, "consensus-dynamics", f"sbm-{paramspace_matrix_weighted_sbm.wildcard_pattern}.pdf")
FIG_RANDOM_WALK = j(FIG_DIR, "random-walk", f"sbm-{paramspace_matrix_weighted_sbm.wildcard_pattern}.pdf")

rule all:
    input:
        expand(FIG_CONS_DYN, **params_matrix_weighted_sbm),
        expand(FIG_RANDOM_WALK, **params_matrix_weighted_sbm)

rule run_consensus_dynamics:
    params:
        params = paramspace_matrix_weighted_sbm.instance
    output:
        output_file = RES_CONS_DYN
    script:
        "workflow/run_consensus_dynamics_sbm.py"


rule plot_consensus_dynamics:
    input:
        input_file = RES_CONS_DYN
    output:
        output_file = FIG_CONS_DYN
    script:
        "workflow/plot_consensus_dynamics_sbm.py"


rule run_random_walk:
    params:
        params = paramspace_matrix_weighted_sbm.instance
    output:
        output_file = RES_RANDOM_WALK
    script:
        "workflow/run_random_walk_sbm.py"


rule plot_random_walk:
    input:
        input_file = RES_RANDOM_WALK
    output:
        output_file = FIG_RANDOM_WALK
    script:
        "workflow/plot_random_walk_sbm.py"

