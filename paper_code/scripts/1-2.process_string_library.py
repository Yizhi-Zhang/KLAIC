import pandas as pd

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
site_uid = pd.read_csv('../data/mat_sites_mapping.csv', index_col=0)

string = pd.read_parquet('../data/string_kinase_ppi.parquet')
string = string[string['protein1'].isin(human_pk['UniprotID']) | string['protein2'].isin(human_pk['UniprotID'])]

buljan_kpi = pd.read_excel('../data/kinase_interaction_Buljan.xlsx', skiprows=2)
buljan_kpi = buljan_kpi[buljan_kpi['Bait_id'].isin(human_pk['UniprotID']) | buljan_kpi['Protein_id'].isin(human_pk['UniprotID'])]

# ---------- Process string libraries ----------
def check_string_co_func(row):
    # If SOURCE is Curated or PhosphoSitePlus, return True directly
    if row['SOURCE'] in ['Curated', 'PhosphoSitePlus']:
        return True
    # Check if target_protein is missing
    if pd.isnull(row['target_protein']):
        return False
    return tuple(sorted((row['source'], row['target_protein']))) in string_prot_pairs

for thr in [400, 700]:
    string2 = string[string['combined_score'] >= thr].copy()
    string2 = string2.dropna()
    
    string_prot_pairs = set(
        tuple(sorted((row['protein1'], row['protein2']))) 
        for _, row in string2.iterrows()
    )
    buljan_kpi_pairs = set(
        tuple(sorted((row['Bait_id'], row['Protein_id'])))
        for _, row in buljan_kpi.iterrows()
    )
    string_prot_pairs.update(buljan_kpi_pairs)
    
    for lib in LIBS:
        curatedlib = pd.read_parquet(f'../results/unfiltered/ksr_{lib}.parquet')
        curatedlib['target_protein'] = curatedlib['target'].map(site_uid.set_index('idx')['UID'])
        curatedlib['string_co_func'] = curatedlib.apply(check_string_co_func, axis=1)
        
        curatedlib = curatedlib[curatedlib['mor'] > 0]
        curatedlib = curatedlib[curatedlib['string_co_func'] == True]
        
        curatedlib.to_parquet(f'../results/string{thr}/ksr_{lib}.parquet')
