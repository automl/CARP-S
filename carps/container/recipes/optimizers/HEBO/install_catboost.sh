PIP=$PIP
if [ -z "$PIP" ]; then
    if command -v uv >/dev/null 2>&1; then
        PIP="uv pip"
    else
        PIP="pip"
    fi
fi

$PIP install setuptools wheel jupyterlab conan build --upgrade
CATBOOST_SRC_ROOT="lib/catboost"
git clone https://github.com/catboost/catboost.git $CATBOOST_SRC_ROOT
mkdir -p lib/dists
python -m build \
    --sdist $CATBOOST_SRC_ROOT/catboost/python-package \
    --outdir lib/dists \
    --wheel      # --skip-dependency-check