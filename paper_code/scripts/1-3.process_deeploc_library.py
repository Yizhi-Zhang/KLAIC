import pandas as pd
from utils import add_spatial_label_match_only

LIBS = [
    'curated_kl0.99_top10',
    'curated_kl0.99_top20',
    'curated_kl0.99_top50',
    'curated_kl0.99',
    'curated_kl0.975',
    'curated_kl0.9',
    'curated_networkin',
]

# ---------- Load data ----------
human_pk = pd.read_excel('../data/human_pk.xlsx')
pk_uid2gene_map = dict(zip(human_pk['UniprotID'], human_pk['Gene Names'].str.split(' ').str[0]))

site_uid = pd.read_csv('../data/mat_sites_mapping.csv', index_col=0)

deep_loc = pd.read_excel('../data/deeploc.xlsx')
# ---------- Process hpa location ----------
ksr = pd.read_parquet('../data/final_ksr_curated_kl90.parquet')
ksr2 = pd.read_parquet('../results/unfiltered/ksr_curated_networkin.parquet')
ksr2 = ksr2[ksr2['SOURCE'] == 'NetworKIN']
ksr = pd.concat([ksr, ksr2], ignore_index=True).drop(columns=['mor'])

ksr['source_uid'] = ksr['source'].tolist()
ksr['target_uid'] = ksr['target'].map(site_uid.set_index('idx')['UID'])

all_main_loc = [
    'Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell membrane', 
    'Endoplasmic reticulum', 'Golgi apparatus', 'Lysosome/Vacuole', 'Peroxisome'
]
for col in ['Localizations'] + all_main_loc:
    ksr['source_'+col] = ksr['source_uid'].map(deep_loc.set_index('Protein_ID')[col])
    ksr['target_'+col] = ksr['target_uid'].map(deep_loc.set_index('Protein_ID')[col])

ksr = add_spatial_label_match_only(
    ksr,
    all_main_loc=all_main_loc,
    source_label_col="source_Localizations",
    target_label_col="target_Localizations",
    source_type_col="SOURCE",
    curated_values={"Curated", "PhosphoSitePlus"},
    label_match_col="deeploc_label_match",
)

# ---------- Process deeploc libraries ----------
for lib in LIBS:
    curatedlib = pd.read_parquet(f'../results/unfiltered/ksr_{lib}.parquet')
    curatedlib = pd.merge(curatedlib, ksr, on=['source', 'target', 'SOURCE'], how='left')

    curatedlib = curatedlib[curatedlib['mor'] > 0]
    curatedlib = curatedlib[curatedlib['deeploc_label_match'] == True]
    
    curatedlib.to_parquet(f'../results/deeploc/ksr_{lib}.parquet')
