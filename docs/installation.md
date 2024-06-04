# Installation

To install CARP-S, you can use the following commands:

```bash
git clone https://github.com/AutoML/CARP-S.git
cd CARP-S
conda create -n carps python=3.11
conda activate carps

# Install for usage
pip install .
```

If you want to install CARP-S for development, you can use the following command:
```bash
# Install for development
make install-dev
```

Additionally, you need to install the requirements for the benchmark and optimizer that you 
want to use. For example, if you want to use the `SMAC2.0` optimizer and the `BBOB` benchmark, 
you need to install the requirements for both of them via:

```bash
pip install -r container_recipes/optimizers/SMAC3/SMAC3_requirements.txt
pip install -r container_recipes/benchmarks/BBOB/BBOB_requirements.txt
```

For some benchmarks, it might additionally be necessary to download data, 
such as the surrogate models, in order to run the benchmark. For example for YAHPO, you can
download the surrogate benchmarks with
```bash
mkdir data; cd data; git clone https://github.com/slds-lmu/yahpo_data
```
