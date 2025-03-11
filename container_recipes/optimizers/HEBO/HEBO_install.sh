PIP=$PIP

if [ -z "$PIP" ]
then
    PIP="pip"
fi

hebo_root="lib/HEBO"

# git clone https://github.com/huawei-noah/HEBO.git $hebo_root

> $hebo_root/HEBO/requirements.txt
$PIP install scipy --upgrade
$PIP install numpy pandas pymoo scikit-learn --upgrade
$PIP install gpytorch
$PIP install GPy
$PIP install numpy --upgrade

# Install Catboost
if [ ! -f "lib/dists/catboost-1.2.7.tar.gz" ]; then
    echo "catboost source file does not exist at 'lib/dists/catboost-1.2.7.tar.gz'. Installing catboost from source (can take a while)."
    . $hebo_root/install_catboost.sh
fi
$PIP install lib/dists/catboost-1.2.7.tar.gz

# $PIP install setuptools wheel jupyterlab conan --upgrade

# git clone https://github.com/catboost/catboost.git $CATBOOST_SRC_ROOT
# $PIP install $CATBOOST_SRC_ROOT/catboost/python-package


# Install rest of the dependencies
$PIP install disjoint-set
$PIP install -e lib/HEBO/HEBO
$PIP install numpy --upgrade
$PIP install scipy --upgrade