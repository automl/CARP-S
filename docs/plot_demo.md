# Plot Demo
This file demonstrates the analysis capabilities from carps.
For the subset data, it shows which functions generate the plots.
Alternatively, to create the plots, you can also run
```bash
python -m carps.analysis.generate_report \
    --result_path=../results/subsets/logs_normalized.parquet \
    --report_name=report \
    --normalize_results=False
```
from the commandline.

The next cell loads the results and preprocesses them. The resulting dataframe has the following columns:
```yaml
# Identifying information of the run
optimizer_id: The identifier of the optimizer.
task_id: The identifier of the task.
seed: Which seed has been used. The combination of a unique `optimizer_id`, `task_id`, `seed`, equals one optimization run.
experiment_id: Experiment id. Enumeration of all runs.

# Information about the progress of the optimization
n_trials: The number of trials that have been evaluated so far.
n_function_calls: The number of times the objective function has been called. This can differ from `n_trials`, when we are in the multi-fidelity settings and can call the objective function with a lower fidelity. This still results in a full increase of `n_function_calls`, but only a fractional increase in `n_trials`.
n_trials_norm: The number of trials, normalized per run.
time: The elapsed time.
time_norm: The elapsed time, normalized per run. 

# Information about the trial (ask)
trial_info__config: The configuration that has been evaluated.
trial_info__instance: The instance that the configuration has been evaluated on (None for anything but Algorithm Configuration).
trial_info__seed: The seed with which the objective function has been evaluated. In case of stochastic objective functions, we might wish to evaluate the objective function several times with different seeds for the same configuration.
trial_info__budget: The multi-fidelity resource, e.g. the number of epochs. None for anything but multi-fidelity.
trial_info__normalized_budget: The normalized budget (normalized by the maximum budget as indicated by the task).
trial_info__name: An optional name for the trial.
trial_info__checkpoint: An optional checkpoint for the trial.

# Information about the evaluated trial (tell)
## Cost related
trial_value__cost: The objective function value (lower is better).
trial_value__cost_raw: Same as `trial_value__cost`.
trial_value__cost_norm: Normalized cost, min and max are taken over all runs for one task.

## The incumbent cost
trial_value__cost_inc: The incumbent cost (best/lowest cost seen so far).
trial_value__cost_inc_norm: Incumbent cost, normalized over all runs for one task.
trial_value__cost_inc_norm_log: Logarithm of incumbent cost.

### Multi-objective
hypervolume: The hypervolume as calculated over all runs for one task.
reference_point: Reference point as determined over all runs for one task (worst seen combo).

## Time related
trial_value__time: The time the objective function took to evaluate.
trial_value__virtual_time: If the objective function is a surrogate, it can still return an evaluation time. This is marked then as virtual time.
trial_value__status: The status of the trial. Mostly hopefully `SUCCESS`.
trial_value__starttime: The start time of the objective function evaluation.
trial_value__endtime: The end time of the objective function evaluation.

# Information about the task
benchmark_id: The identifier of the benchmark collection the task belongs to.
task_type: The task type, either blackbox, multi-fidelity, multi-objective, or multi-fidelity-objective / momf.
subset_id: The subset id, mostly `None`, `dev` or `test`.
set: Another name for `subset_id`.
task.optimization_resources.n_trials: The optimization resources for the tasks in terms of number of trials.
```

## Download Results
Optionally download results from huggingface.
Prerequisite: Install `huggingface`.


```python
from huggingface_hub import hf_hub_download

repo_id = "benjamhc/carps"
dataset_name = "logs_normalized.parquet"  # ~ 1GB
handle = hf_hub_download(repo_id=repo_id, filename=dataset_name, repo_type="dataset")

```

## Generate Plots
You can call the following function to automatically generate all plots shown below:
```python
from carps.analysis.generate_report import generate_report

generate_report(
    result_path=handle,
    report_dir="reports",
    report_name="subselection,
    normalize_results=False,  # Result file is already normalized
)
```
This generates all plots under `<report_dir>/<report_name>/figures`.

## Load Results


```python
from carps.analysis.generate_report import load_results

figure_dir = "figures"

# Load results from file, they are already normalized in this case
results_full = load_results(result_path=handle, normalize=False)

# We only plot subset blackbox test
results = results_full[(results_full["task_type"]=="blackbox") & (results_full["set"]=="test")]
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[11:39:49] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Loading results from                                                    <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">generate_report.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py#1028" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1028</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #800080; text-decoration-color: #800080">/home/numina/.cache/huggingface/hub/datasets--benjamhc--carps/snapshots</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #800080; text-decoration-color: #800080">/bbd9164e92011a752c62034dddb9ecdf7872f1ed/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">logs_normalized.parquet</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[11:40:13] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Preprocessing results                                                   <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">generate_report.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py#1032" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1032</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[11:40:14] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Columns: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Index</span><span style="font-weight: bold">([</span><span style="color: #008000; text-decoration-color: #008000">'task_id'</span>, <span style="color: #008000; text-decoration-color: #008000">'optimizer_id'</span>, <span style="color: #008000; text-decoration-color: #008000">'seed'</span>, <span style="color: #008000; text-decoration-color: #008000">'level_3'</span>,           <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">generate_report.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py#1033" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1033</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'n_trials'</span>,                                                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'n_function_calls'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_info__config'</span>,                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'trial_info__instance'</span>,                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_info__seed'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_info__budget'</span>,                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_info__normalized_budget'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_info__name'</span>,             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_info__checkpoint'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost'</span>,                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'trial_value__time'</span>,                                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__virtual_time'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__status'</span>,              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__starttime'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__endtime'</span>,                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'benchmark_id'</span>,                                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'task_type'</span>, <span style="color: #008000; text-decoration-color: #008000">'subset_id'</span>,                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'task.optimization_resources.n_trials'</span>,                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_raw'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_inc'</span>, <span style="color: #008000; text-decoration-color: #008000">'time'</span>,        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'experiment_id'</span>, <span style="color: #008000; text-decoration-color: #008000">'n_trials_norm'</span>, <span style="color: #008000; text-decoration-color: #008000">'hypervolume'</span>,                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'reference_point'</span>,                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'minimum_cost'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_norm'</span>,                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_log'</span>,                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_inc_log'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_log_norm'</span>,       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_inc_log_norm'</span>, <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_inc_norm'</span>,  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                <span style="color: #008000; text-decoration-color: #008000">'trial_value__cost_inc_norm_log'</span>, <span style="color: #008000; text-decoration-color: #008000">'time_norm'</span><span style="font-weight: bold">]</span>,                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>               <span style="color: #808000; text-decoration-color: #808000">dtype</span>=<span style="color: #008000; text-decoration-color: #008000">'object'</span><span style="font-weight: bold">)</span>                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                       </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> <span style="color: #808000; text-decoration-color: #808000">...</span>skipping normalization as requested                                  <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">generate_report.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/generate_report.py#1044" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1044</span></a>
</pre>



## Critical Difference Plot
We analyze experimental results from different viewpoints. Most importantly, we aggregate the results via
rankings. We use the library autorank (Herbold, 2020) for determining the ranks and critical differences.
The ranking is performed on the raw performance values, averaged across seeds. To be more precise, we
use the frequentist approach (Demšar, 2006): We use the non-parametric Friedman test as an omnibus test
to determine whether there are any significant differences between the median values of the populations.
We use this test because we have more than two populations, which cannot be assumed to be normally
distributed. We use the post hoc Nemenyi test to infer which differences are significant. The significance level
is α = 0.05. In order to be considered different, the difference between the mean ranks of two optimizers must
be greater than the critical difference.


```python
from carps.analysis.generate_report import plot_critical_difference

resulting_files_critical_difference = plot_critical_difference(results, output_dir=figure_dir, show_figure=True)
```


    
![png](plot_demo_files/plot_demo_7_0.png)
    


## Performance per Task
We visualize the final performance per task as a heatmap with tasks as rows and optimizers as columns.
The cells display the raw final performance, averaged over seeds per task, and the colormap indicates 
how well an optimizer performed in comparison. The colormap is using the normalized performance values.


```python
from carps.analysis.generate_report import plot_performance_per_task

resulting_files_performance_per_task = plot_performance_per_task(results, output_dir=figure_dir, replot=True, show_figure=True)
```


    
![png](plot_demo_files/plot_demo_9_0.png)
    


## Final Performance Boxplot
This is a boxplot together with a violin plot, showing the raw (but normalized) distribution of the results.
The optimizers are sorted by their median value to match the critical difference assessment.


```python
from carps.analysis.generate_report import plot_finalperfboxplot

resulting_files_finalperfboxplot = plot_finalperfboxplot(results, output_dir=figure_dir, replot=True, show_figure=True)
```


    
![png](plot_demo_files/plot_demo_11_0.png)
    


## Performance over Time
We can inspect the anytime performance in two ways. 
The first is visualizing the incumbent cost over iterations: Either aggregated and normalized (and interpolated), or per task.
The caveat of the first method is that we cannot distinguish well between optimizers.
Thus, we normally resort to the ranking over time as determined via statistical testing.

### Aggregated over tasks, normalized
The plot shows the mean incumbent cost over iterations (both normalized and interpolated) with 95%-CI.
Mostly indistinguishable thus not advised.


```python
from carps.analysis.generate_report import plot_performance_over_time

%matplotlib inline

resulting_files_perfovertime = plot_performance_over_time(
    results, output_dir=figure_dir, per_task=False, replot=True, show_figure=True
)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[11:41:21] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Create dataframe for neat plotting by aligning x-axis <span style="color: #800080; text-decoration-color: #800080">/</span> interpolating        <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/gather_data.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">gather_data.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/gather_data.py#718" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">718</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         budget.                                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                  </span>
</pre>




    
![png](plot_demo_files/plot_demo_13_1.png)
    


### Per Task
⏳ This might take a while because this uses seaborn's grid plot.


```python
from carps.analysis.generate_report import plot_performance_over_time

%matplotlib inline

resulting_files_perfovertime = plot_performance_over_time(
    results, output_dir=figure_dir, per_task=True, replot=True, show_figure=True
)
```


    
![png](plot_demo_files/plot_demo_15_0.png)
    


### Ranking
The ranking is calculated per step in the same way as for the critical difference diagram, via statistical testing.

⏳ This might take a while...


```python
from carps.analysis.generate_report import plot_ranks_over_time

%matplotlib inline

resulting_files_rank_over_time = plot_ranks_over_time(results, output_dir=figure_dir, replot=True, show_figure=True)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[11:42:02] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Create dataframe for neat plotting by aligning x-axis <span style="color: #800080; text-decoration-color: #800080">/</span> interpolating        <a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/gather_data.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">gather_data.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/numina/Documents/repos/CARP-S-Experiments/lib/CARP-S/carps/analysis/gather_data.py#718" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">718</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         budget.                                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                  </span>
</pre>




    
![png](plot_demo_files/plot_demo_17_1.png)
    

