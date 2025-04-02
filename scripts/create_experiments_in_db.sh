# BLACKBOX
BB_OPTIMIZERS=(
    "+optimizer/randomsearch=config"
    "+optimizer/Ax=config"
    "+optimizer/hebo=config"
    "+optimizer/nevergrad=bayesopt"
    "+optimizer/nevergrad=Hyperopt"
    "+optimizer/nevergrad=NoisyBandit"
    "+optimizer/nevergrad=DE"
    "+optimizer/nevergrad=ES"
    "+optimizer/optuna=SO_TPE"
    "+optimizer/smac20=blackbox"
    "+optimizer/smac20=hpo"
    "+optimizer/synetune=BO"
    "+optimizer/synetune=BORE"
    "+optimizer/synetune=KDE"
    "+optimizer/synetune=BO_RS"
    "+optimizer/synetune=MOREA"
    "+optimizer/scikit_optimize=BO_GP_EI"
    "+optimizer/scikit_optimize=BO_GP_LCB"
    "+optimizer/scikit_optimize=BO_GP_PI"
    "+optimizer/scikit_optimize=BO"
)

for OPT in "${BB_OPTIMIZERS[@]}"; do
    echo $OPT
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/blackbox/dev=glob(*)' 'seed=range(1,21)' -m 
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/blackbox/test=glob(*)' 'seed=range(1,21)' -m 
done

# MULTIFIDELITY
MF_OPTIMIZERS=(
    "+optimizer/randomsearch=config"
    "+optimizer/dehb=multifidelity"
    "+optimizer/smac20=hyperband"
    "+optimizer/smac20=multifidelity"
    "+optimizer/synetune=DEHB"
    "+optimizer/synetune=SyncMOBSTER"
)
for OPT in "${MF_OPTIMIZERS[@]}"; do
    echo $OPT
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multifidelity/dev=glob(*)' 'seed=range(1,21)' -m 
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multifidelity/test=glob(*)' 'seed=range(1,21)' -m 
done



# MULTI-OBJECTIVE
MO_OPTIMIZERS=(
    # "+optimizer/randomsearch=config"
    # "+optimizer/nevergrad=cmaes"
    # "+optimizer/nevergrad=DE"
    "+optimizer/nevergrad=ES"
    "+optimizer/optuna=MO_NSGAII"
    "+optimizer/optuna=MO_TPE"
    "+optimizer/smac20=multiobjective_gp"
    "+optimizer/smac20=multiobjective_rf"
    "+optimizer/synetune=BO_MO_LS"
    "+optimizer/synetune=BO_MO_RS"
    "+optimizer/synetune=MOREA"
)

for OPT in "${MO_OPTIMIZERS[@]}"; do
    echo $OPT

    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multiobjective/dev=glob(*)' 'seed=range(1,21)' -m
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multiobjective/test=glob(*)' 'seed=range(1,21)' -m
done


# MOMF
MOMF_OPTIMIZERS=(
    "+optimizer/randomsearch=config"
    "+optimizer/smac20=momf_gp"
    "+optimizer/smac20=momf_rf"
    "+optimizer/nevergrad=cmaes"
)
for OPT in "${MOMF_OPTIMIZERS[@]}"; do
    echo $OPT

    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multifidelityobjective/dev=glob(*)' 'seed=range(1,21)' -m
    python -m carps.container.create_cluster_configs +cluster=local $OPT '+task/subselection/multifidelityobjective/test=glob(*)' 'seed=range(1,21)' -m
done