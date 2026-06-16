import pandas as pd
import numpy as np

res_merged = pd.DataFrame()
for filter in ['unfiltered', 'deeploc', 'string400', 'string700']:
    res = pd.read_parquet(f'../results/{filter}/benchmark_results.parquet')
    # res = pd.read_parquet(f'../results/{filter}/benchmark_results_same_kinase.parquet')
    res = res[[
        'method', 'n_Detected', 'n_Overlap', 'total_event',
        'Phit@0.1', 'Phit@0.15', 'Phit@0.2', 
        'macroRecall@0.1', 'macroRecall@0.15', 'macroRecall@0.2',
        'Rank (paper)', 'AUROC', 
        'direction_acc', 'direction_acc_pos', 'direction_acc_neg']]
    res = res.rename(columns={
        'method': 'Library',
        'n_Detected': 'n_All_kinase',
        'n_Overlap': 'n_Evaluable_kinase',
        'total_event': 'n_Evaluable_events',
        'macroRecall@0.1': 'Recall@0.1',
        'macroRecall@0.15': 'Recall@0.15',
        'macroRecall@0.2': 'Recall@0.2',
        'Rank (paper)': 'Percentile_rank',
        'direction_acc': 'Direction_accuracy',
        'direction_acc_pos': 'Direction_accuracy_activation',
        'direction_acc_neg': 'Direction_accuracy_inhibition',
    })
    res['AUROC_median'] = [np.median(x) for x in res['AUROC']]
    res['Percentile_rank_median'] = [np.median(x) for x in res['Percentile_rank']]
    res['Filter'] = filter
    res = res[[
        'Filter', 'Library', 
        'n_All_kinase', 'n_Evaluable_kinase', 'n_Evaluable_events',
        'AUROC_median', 
        'Percentile_rank_median', 
        
        'Phit@0.1', 'Phit@0.15', 'Phit@0.2', 
        'Recall@0.1', 'Recall@0.15', 'Recall@0.2',
        'Direction_accuracy', 'Direction_accuracy_activation', 'Direction_accuracy_inhibition',
    ]]

    res_merged = pd.concat([res_merged, res])

res_merged = res_merged[
    ~((res_merged['Filter'] != 'unfiltered') & (res_merged['Library'] == 'curated'))
]

res_merged.to_csv('../results/benchmark_results.csv', index=False)
# res_merged.to_csv('../results/benchmark_results_same_kinase.csv', index=False)