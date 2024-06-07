folder=$1 # folder to run stuff in
n_tasks=$2  # number of points
ks=$3  # subset sizes as comma separated list

cd $folder

rm subset*
rm log*
rm info.csv
rm df_crit.txt

echo "Convert df_crit.csv to points"
python3 ../extract_csv.py df_crit.csv df_crit.txt
echo "Subselect for $ks sizes"
../run.sh ../a.out df_crit.txt 3 $n_tasks $ks
echo "Format"
../format.sh df_crit.csv $ks
echo "Prepare selection for complement subset"
echo "Run again"
../again.sh ../a.out df_crit.csv $n_tasks 3 $ks
echo "Gather info"
../extract.sh $ks >> info.csv