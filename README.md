# Matrix-Weighted Networks

- [Data](https://drive.google.com/drive/folders/1_DZEIUtMn-m-C9ub1EzMyZPd9m2F8tsx?usp=sharing)
- [Fig](https://drive.google.com/drive/folders/1tcMWxp7pGWC95DWVJBVAFcNC5IFLgxgx?usp=sharing)

## Paper
- [arXiv](https://arxiv.org/abs/2410.05188)

To cite our work, please use the following BibTeX entry:
```bibtex
@article{tian2024matrix,
        title        = {Matrix-weighted networks for modeling multidimensional dynamics},
        author       = {Yu Tian and Sadamori Kojaku and Hiroki Sayama and Renaud Lambiotte},
        year         = 2024,
        publisher    = {{arXiv}},
        number       = {arXiv:2410.05188},
        eprint       = {arXiv:2410.05188},
        archiveprefix = {arxiv},
}
```

## Reproducing Our Results

### Setup

1. Set up the virtual environment and install the required packages:
```bash
conda create -n matnet python=3.10
conda activate matnet
conda install -c conda-forge mamba -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy==1.23.5 numba scipy pandas networkx seaborn matplotlib ipykernel tqdm black -y
```

2. Create a file `config.yaml` with the following content and place it under the `workflow` folder:
```yaml
data_dir: "data/"
fig_dir: "figs/"
```

### Run Simulation

Run the following command to execute the `Snakemake` workflow:
```bash
snakemake --cores 24 all
```
This will generate all files needed to produce the figures. Then, run
```bash
snakemake --cores 24 figs
```
You can change the number of cores to use, instead of 24.
