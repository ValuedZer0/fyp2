import pandas as pd
from configs import OUTLIER_METHODS, NORM_METHODS, DISTANCE_METRICS
from run_single import run_single_config

results_list = []
for outlier in OUTLIER_METHODS:
    for norm in NORM_METHODS:
        for metric in DISTANCE_METRICS:
            print(f"Running {outlier}, {norm}, {metric} ...")
            agg = run_single_config('iris', outlier, norm, metric, n_runs=30)
            row = {'outlier': outlier, 'norm': norm, 'metric': metric}
            for k, v in agg.items():
                row[f'{k}_mean'] = v[0]
                row[f'{k}_std'] = v[1]
                row[f'{k}_min'] = v[2]
                row[f'{k}_max'] = v[3]
            results_list.append(row)

df = pd.DataFrame(results_list)
df.to_csv('results/iris_results.csv', index=False)
print("Done! Saved to results/iris_results.csv")