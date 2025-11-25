# Plot Demos

This page demonstrates the analysis capabilities from carps. For the subset data, it shows which functions generate the plots.
Alternatively, to create the plots, you can also run

```bash
python -m carps.analysis.generate_report \
    --result_path=../results/subsets/logs_normalized.parquet \
    --report_name=report \
    --normalize_results=False
```

## Critical Difference Plot
We analyze experimental results from different viewpoints. Most importantly, we aggregate the results via rankings. We use the library autorank (Herbold, 2020) for determining the ranks and critical differences. The ranking is performed on the raw performance values, averaged across seeds. To be more precise, we use the frequentist approach (Demšar, 2006): We use the non-parametric Friedman test as an omnibus test to determine whether there are any significant differences between the median values of the populations. We use this test because we have more than two populations, which cannot be assumed to be normally distributed. We use the post hoc Nemenyi test to infer which differences are significant. The significance level is α = 0.05. In order to be considered different, the difference between the mean ranks of two optimizers must be greater than the critical difference.

```python
from carps.analysis.generate_report import plot_critical_difference

resulting_files_critical_difference = plot_critical_difference(results, output_dir=figure_dir, show_figure=True)
```

![Critical Difference](images/plots/figures/critical_difference.png)

## Performance per Task
We visualize the final performance per task as a heatmap with tasks as rows and optimizers as columns. The cells display the raw final performance, averaged over seeds per task, and the colormap indicates how well an optimizer performed in comparison. The colormap is using the normalized performance values.

```python
from carps.analysis.generate_report import plot_performance_per_task

resulting_files_performance_per_task = plot_performance_per_task(results, output_dir=figure_dir, replot=True, show_figure=True)
```

![Performance per Task](images/plots/figures/performance_per_task.png)

## Final Performance Boxplot
This is a boxplot together with a violin plot, showing the raw (but normalized) distribution of the results. The optimizers are sorted by their median value to match the critical difference assessment.

```python
from carps.analysis.generate_report import plot_finalperfboxplot

resulting_files_finalperfboxplot = plot_finalperfboxplot(results, output_dir=figure_dir, replot=True, show_figure=True)

```

![Final Performance Boxplot](images/plots/figures/final_performance.png)

## Performance over Time
We can inspect the anytime performance in two ways. The first is visualizing the incumbent cost over iterations: Either aggregated and normalized (and interpolated), or per task. The caveat of the first method is that we cannot distinguish well between optimizers. Thus, we normally resort to the ranking over time as determined via statistical testing.

### Aggregated over tasks, normalized
The plot shows the mean incumbent cost over iterations (both normalized and interpolated) with 95%-CI. Mostly indistinguishable thus not advised.

```python
from carps.analysis.generate_report import plot_performance_over_time

%matplotlib inline

resulting_files_perfovertime = plot_performance_over_time(
    results, output_dir=figure_dir, per_task=False, replot=True, show_figure=True
)
```

![aggregated over tasks](images/plots/figures/aggregated_over_task.png)

### Per Task
⏳ This might take a while because this uses seaborn's grid plot.

```python
from carps.analysis.generate_report import plot_performance_over_time

%matplotlib inline

resulting_files_perfovertime = plot_performance_over_time(
    results, output_dir=figure_dir, per_task=True, replot=True, show_figure=True
)
```

![Per Task](images/plots/figures/per_task.png)

### Ranking
The ranking is calculated per step in the same way as for the critical difference diagram, via statistical testing.

⏳ This might take a while...

```python
from carps.analysis.generate_report import plot_ranks_over_time

%matplotlib inline

resulting_files_rank_over_time = plot_ranks_over_time(results, output_dir=figure_dir, replot=True, show_figure=True)
```

![Ranks Over Time](images/plots/figures/ranking.png)