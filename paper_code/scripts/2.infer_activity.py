import pandas as pd
from utils import run_zscore

expr_mat = pd.read_csv('../data/mat_human_pk.csv', index_col=0)

LIBS = [
    'curated',
    'curated_kl0.99_top10',
    'curated_kl0.99_top20',
    'curated_kl0.99_top50',
    'curated_kl0.99',
    'curated_kl0.975',
    'curated_kl0.95',
    'curated_kl0.9',
    'curated_networkin'
]

FILTERS = [
    'unfiltered',
    'string700',
    'string400',
    'deeploc'
]

for filter in FILTERS:
    for lib in LIBS:
        if filter != 'unfiltered' and lib == 'curated':
            continue
        
        network = pd.read_parquet(f'../results/{filter}/ksr_{lib}.parquet')
        act = run_zscore(expr_mat, network, minsize=5)
        act.to_parquet(f'../results/{filter}/zscore_{lib}.parquet')
 