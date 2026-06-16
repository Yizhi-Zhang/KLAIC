import pandas as pd
import numpy as np
from utils import compute_recall_at_k, run_phit, run_rank, run_perturb_bench, cal_direction_acc, cal_direction_acc2

# ---------- Load data ----------
human_pk = pd.read_excel('../data/human_pk.xlsx')

meta = pd.read_csv('../data/meta_human_pk.csv', index_col=0).rename(columns={'target':'gene', 'UniprotID':'target'})
meta_sign = meta[['id', 'sign']].copy().drop_duplicates()
meta_sign.index = meta_sign['id'].tolist()

# ---------- Run evaluation ----------
for folder in ['unfiltered', 'deeploc', 'string400', 'string700']:
    try:
        print(f"Running {folder}...")

        path_dict = {
            'curated': f'../3.results/{folder}/zscore_curated.parquet',
            
            'curated_kl0.99_top10': f'../3.results/{folder}/zscore_curated_kl0.99_top10.parquet',
            'curated_kl0.99_top20': f'../3.results/{folder}/zscore_curated_kl0.99_top20.parquet',
            'curated_kl0.99_top50': f'../3.results/{folder}/zscore_curated_kl0.99_top50.parquet',
            'curated_kl0.99': f'../3.results/{folder}/zscore_curated_kl0.99.parquet',
            'curated_kl0.975': f'../3.results/{folder}/zscore_curated_kl0.975.parquet',
            'curated_kl0.95': f'../3.results/{folder}/zscore_curated_kl0.95.parquet',
            'curated_kl0.9': f'../3.results/{folder}/zscore_curated_kl0.9.parquet',

            'curated_networkin': f'../3.results/{folder}/zscore_curated_networkin.parquet',
        }

        zscores_dict = {}
        for lib, path in path_dict.items():
            zscore = pd.read_parquet(path)
            
            # Multiply z-score columns by their sign to get signed z-scores
            sign_series = meta_sign.set_index("id")["sign"]
            common_cols = zscore.columns.intersection(sign_series.index)
            zscore_signed = zscore.copy()
            zscore_signed[common_cols] = zscore_signed[common_cols] * sign_series[common_cols]
            
            zscores_dict[lib] = (zscore, zscore_signed)
        
        results = {}

        for method, (act_score_df, act_score_signed_df) in zscores_dict.items():
            row = {}  # Place all results for this method in this dict

            # Filter rows where index is in human_pk['UniprotID']
            # Remove rows and columns which are all NaN
            act_score_signed_df = act_score_signed_df[act_score_signed_df.index.isin(human_pk['UniprotID'])]
            act_score_signed_df = act_score_signed_df.loc[~act_score_signed_df.isna().all(axis=1), ~act_score_signed_df.isna().all()]
            
            act_score_df = act_score_df[act_score_df.index.isin(human_pk['UniprotID'])]
            act_score_df = act_score_df.loc[~act_score_df.isna().all(axis=1), ~act_score_df.isna().all()]

            # Number of detected targets in this method's result
            row['n_Detected'] = len(set(act_score_df.index))
            # Number of overlaps between detected targets and all targets in meta
            row['n_Overlap'] = len(set(act_score_df.index) & set(meta['target']))

            if row['n_Overlap'] == 0:
                for topk in [0.1, 0.15, 0.2]:
                    row[f'Phit@{topk}'] = np.nan
                row['Rank (paper)'] = np.nan
                row['AUROC'] = np.nan
                row['direction_acc'] = np.nan
                row['direction_acc_pos'] = np.nan
                row['direction_acc_neg'] = np.nan

            else:   
                # ---- Recall & Phit (scalar) ----
                for topk in [0.1, 0.15, 0.2]:
                    row[f'macroRecall@{topk}'] = float(compute_recall_at_k(meta, act_score_signed_df, k=topk, mode="macro", metric='scaled_rank')['recall_at_k'])
                    row[f'Phit@{topk}']        = float(run_phit(act_score_signed_df, meta, k=topk, average=True, metric='scaled_rank'))
            
                # ---- Rank (list) ----
                rank_tbl = run_rank(act_score_signed_df, meta, average=True)
                row['Rank (paper)'] = rank_tbl['scaled_rank'].tolist() if len(rank_tbl)>0 else np.nan
                row['n_Overlap'] = len(rank_tbl)

                if row['n_Overlap'] > 0:
                    # ---- AUROC (list) ----
                    out = run_perturb_bench(act_score_df, meta, scale_data=True, rm_bg=False, n_iter=1000, random_state=42)
                    row['AUROC'] = list(out) if not isinstance(out, list) else out
                else:
                    row['AUROC'] = np.nan
                
                # ---- direction_acc (scalar) ----
                row['direction_acc'], row['wrong_list'], row['total_event'] = cal_direction_acc(meta, act_score_df)
                row['direction_acc_pos'], row['direction_acc_neg'] = cal_direction_acc2(meta, act_score_df)

            results[method] = row

        # Build all results at once; columns that contain lists will be of object dtype
        res_df = pd.DataFrame.from_dict(results, orient='index')
        res_df = res_df.reset_index().rename(columns={'index': 'method'})

        res_df.to_parquet(f'../3.results/{folder}/benchmark_kl_results.parquet')

        print(f"Done {folder}.")
    
    except Exception as e:
        print(f"Error in {folder}: {e}")
        continue