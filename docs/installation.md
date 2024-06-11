# Installation

### Installation from PyPI

To install CARP-S, you can simply use `pip`:

```bash
conda create -n carps python=3.11
conda activate carps
pip install carps
```

Additionally, you need to install the requirements for the benchmark and optimizer that you want to use.
For example, if you want to use the `SMAC2.0` optimizer and the `BBOB` benchmark, you need to install the
requirements for both of them via:

```bash
pip install carps[smac,bbob]
```

All possible install options for benchmarks are:
```bash
dummy,bhob,hpob,mfpbench,pymoo,yahpo
```

All possible install options for optimizers are:
```bash
dummy,dehb,hebo,nevergrad,optuna,skopt,smac,smac14,synetune
```

Please note that installing all requirements for all benchmarks and optimizers in a single 
environment will not be possible due to conflicting dependencies.

### Installation from Source

If you want to install from source, you can clone the repository and install CARP-S via:

```bash
git clone https://github.com/AutoML/CARP-S.git
cd CARP-S
conda create -n carps python=3.11
conda activate carps

# Install for usage
pip install .
```

For installing the requirements for the optimizer and benchmark, you can then use the following command:
```bash
pip install ".[smac,bbob]"
```

If you want to install CARP-S for development, you can use the following command:
```bash
make install-dev
```

### Additional Steps for Benchmarks

For HPOBench, it is necessary to install the requirements via:
```bash
bash container_recipes/benchmarks/HPOBench/install_HPOBench.sh
```

For some benchmarks, it is necessary to download data, 
such as surrogate models, in order to run the benchmark: 

-   For HPOB, you can download the surrogate benchmarks with
    ```bash
    bash container_recipes/benchmarks/HPOB/download_data.sh
    ```

-   For MFPBench, you can download the surrogate benchmarks with
    ```bash
    bash container_recipes/benchmarks/MFPBench/download_data.sh
    ```

-   For YAHPO, you can download the required surrogate benchmarks and meta-data with
    ```bash
    bash container_recipes/benchmarks/YAHPO/prepare_yahpo.sh
    ```