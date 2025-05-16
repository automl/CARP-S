PIP=$PIP
if [ -z "$PIP" ]
then
    PIP="pip"
fi

$PIP install setuptools wheel jupyterlab conan build --upgrade
CATBOOST_SRC_ROOT="lib/catboost"
git clone https://github.com/catboost/catboost.git $CATBOOST_SRC_ROOT
mkdir -p lib/dists
python -m build \
    --sdist $CATBOOST_SRC_ROOT/catboost/python-package \
    --outdir lib/dists \
    --wheel      # --skip-dependency-check