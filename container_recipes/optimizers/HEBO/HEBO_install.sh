PIP=$PIP

if [ -z "$PIP" ]
then
    PIP="pip"
fi

# git clone https://github.com/huawei-noah/HEBO.git lib/HEBO

# Read lines of requirements.txt and install each package
while read -r line; do
    # Strip line of anything after a = > or <
    line=$(echo $line | sed -e 's/[<=>].*//')
    echo $line >> lib/HEBO/HEBO/requirements_tmp.txt
done < lib/HEBO/HEBO/requirements.txt

# Replace the requirements.txt with the stripped version
mv lib/HEBO/HEBO/requirements_tmp.txt lib/HEBO/HEBO/requirements.txt

$PIP install -e lib/HEBO/HEBO