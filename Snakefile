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

params_consensus_dynamics = {
    "n_nodes": [120],
    "dim": [2, 3, 5, 10],
    "pin": [0.1, 0.3],
    "pout": [0.1, 0.3],
    "noise": [0, 0.1, 0.3],
    "coherence": [1, 0.8],
    "n_communities": [2, 3, 4]
}
paramspace_consensus_dynamics = to_paramspace(params_consensus_dynamics)

RES_CONS_DYN = j(DATA_DIR, "consensus-dynamics", f"sbm-{paramspace_consensus_dynamics.wildcard_pattern}.npz")
FIG_CONS_DYN = j(FIG_DIR, "consensus-dynamics", f"sbm-{paramspace_consensus_dynamics.wildcard_pattern}.pdf")

rule all:
    input:
        expand(FIG_CONS_DYN, **params_consensus_dynamics)

rule run_consensus_dynamics:
    params:
        params = paramspace_consensus_dynamics.instance
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

