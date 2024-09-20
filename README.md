# Matrix-Weighted Networks

- [Data](https://drive.google.com/drive/folders/1tptfbIBAlyO12O3dvFMStsz8NBTKo9Nc?usp=drive_link)
- [Fig](https://drive.google.com/drive/folders/1htAwu0tlJHD1BfpMdNmUcHdWIPS0yO53?usp=drive_link)

## Paper
- [arXiv](...)

```
Yu Tian, Sadamori Kojaku, Hiroki Sayama, Renaud Lambiotte. "Matrix-Weighted Networks", ...
```

To cite our work, please use the following BibTeX entry:
```bibtex
@article{Tian2024MatrixWeightedNetworks,
        title        = {Matrix-Weighted Networks},
        author       = {Yu Tian and Sadamori Kojaku and Hiroki Sayama and Renaud Lambiotte},
        year         = 2024,
        publisher    = {{arXiv}},
        number       = {arXiv:...},
        eprint       = {arXiv:...},
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

2. Install the in-house packages

```bash
cd libs/matnet && pip install -e .
```

4. Create a file `config.yaml` with the following content and place it under the `workflow` folder:
```yaml
data_dir: "data/"
fig_dir: "figs/"
```

Note that the script will generate over 1T byte of data under this `data/` folder. Make sure you have sufficient disk space.

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