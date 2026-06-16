import pandas as pd

# ---------- Load data ----------
human_pk = pd.read_excel('../data/human_pk.xlsx')
pk_uid2gname = human_pk.set_index('UniprotID')['Gene Names'].str.split(' ').str[0].to_dict()
pk_gname2uid = {v:k for k,v in pk_uid2gname.items()}

ksr_lib = pd.read_parquet('../data/final_ksr_curated_kl90.parquet')
ksr_lib = ksr_lib[ksr_lib['source'].isin(human_pk['UniprotID'])]

# ---------- Process unfiltered libraries ----------
FILTER_TYPE = 'unfiltered'

## 1. Curated
curatedlib = ksr_lib[ksr_lib['SOURCE'] != 'KinaseLibrary'].copy()
curatedlib.to_parquet(f'../results/{FILTER_TYPE}/ksr_curated.parquet')

## 2. KL-expanded
for thr in [0.9, 0.95, 0.975, 0.99]:
    curatedlib = ksr_lib[ksr_lib['mor'] > thr].copy()
    curatedlib.to_parquet(f'../results/{FILTER_TYPE}/ksr_curated_kl{thr}.parquet')

for top in [10, 20, 50]:
    curatedlib = ksr_lib[ksr_lib['mor'] > 0.99].copy()

    def keep_top_kl(df):
        kl = df[df['SOURCE'] == 'KinaseLibrary'].sort_values('mor', ascending=False).head(top)
        return pd.concat([df[df['SOURCE'] != 'KinaseLibrary'], kl])

    curatedlib_filtered = (
        curatedlib.groupby('source', group_keys=False)
        .apply(keep_top_kl)
        .reset_index(drop=True)
    )
    curatedlib = curatedlib_filtered
    curatedlib.to_parquet(f'../results/{FILTER_TYPE}/ksr_curated_kl0.99_top{top}.parquet')

## 3. NetworkIN-expanded
ksr_nw = pd.read_csv('../data/networkin_human_predictions_3.1.tsv', sep='\t')
OUTPUT, SCORE = 'networkin_raw.tsv', 5

# Alias remapping -> HGNC symbol (directly translated from 01.6_prepare_NetworKIN.R)
recode = {
 "ABL":"ABL1","AURORAA":"AURKA","AURORAB":"AURKB","AURORAC":"AURKC","BRK":"PTK6",
 "CAMKIALPHA":"CAMK1","CAMKIIALPHA":"CAMK2A","CAMKIIBETA":"CAMK2B","CAMKIIDELTA":"CAMK2D",
 "CAMKIIGAMMA":"CAMK2G","CAMKIV":"CAMK4","CAMKIDELTA":"CAMK1D","MRCKB":"CDC42BPB",
 "DMPK2":"CDC42BPG","ICK":"CILK1","CK1ALPHA":"CSNK1A1","CK1DELTA":"CSNK1D","CK1EPSILON":"CSNK1E",
 "CK1GAMMA1":"CSNK1G1","CK1GAMMA2":"CSNK1G2","CK1GAMMA3":"CSNK1G3","CK2ALPHA":"CSNK2A1",
 "CK2A2":"CSNK2A2","DNAPK":"PRKDC","GSK3ALPHA":"GSK3A","GSK3BETA":"GSK3B","IKKALPHA":"CHUK",
 "IKKBETA":"IKBKB","IRR":"INSRR","LKB1":"STK11","MAP4K6":"MINK1","MRCKA":"CDC42BPA","RON":"MST1R",
 "MST2":"STK3","TRKC":"NTRK3","PDGFRBETA":"PDGFRB","PDGFRALPHA":"PDGFRA","PDHK1":"PDK1",
 "PDHK2":"PDK2","PDHK3":"PDK3","PDHK4":"PDK4","PKBALPHA":"AKT1","PKBBETA":"AKT2","PKBGAMMA":"AKT3",
 "PKCALPHA":"PRKCA","PKCEPSILON":"PRKCE","PKCETA":"PRKCH","PKCGAMMA":"PRKCG","PKCIOTA":"PRKCI",
 "PKCTHETA":"PRKCQ","PKG1CGKI":"PRKG1","AMPKA1":"PRKAA1","AMPKA2":"PRKAA2","RSK3":"RPS6KA2",
 "P70S6K":"RPS6KB2","LOK":"STK10","MST3":"STK24","YSK1":"STK25","MST4":"STK26","TRKA":"NTRK1",
 "TRKB":"NTRK2","CHAK2":"TRPM6","YES":"YES1","PKAALPHA":"PRKACA","PKABETA":"PRKACB","PKAGAMMA":"PRKACG",
}

df = ksr_nw.copy()
sub_col = "#substrate" if "#substrate" in df.columns else ("substrate" if "substrate" in df.columns else df.columns[0])

df["id"] = df["id"].str.upper().replace(recode)
df["networkin_score"] = pd.to_numeric(df["networkin_score"], errors="coerce")
df = df[df["networkin_score"] >= SCORE].copy()

df["AA"] = df["sequence"].str.extract(r"([a-z]+)", expand=False)          # Phosphorylated residue = lowercase letter
df["target"] = df[sub_col].str.split(" ").str[0]                          # Take the first part of 'substrate'
df["target_site"] = df["target"] + "_" + df["AA"].str.upper() + df["position"].astype(str)

out = df[["id","sequence","target_site"]].rename(columns={"id":"source","target_site":"target"})
out["sequence"] = out["sequence"].str.upper().str.replace("-","_",regex=False)
out["target_protein"] = out["target"].str.split("_").str[0]
out["position"] = out["target"].str.split("_").str[1]
out["mor"] = 1
out = out[["source","target","target_protein","position","mor","sequence"]].drop_duplicates()

ksr_nw = out.copy()
ksr_nw['idx'] = ksr_nw['target']+'|'+ksr_nw['target_protein']+'|'+ksr_nw['position']
ksr_nw.loc[ksr_nw['source'] == 'MST1', 'source'] = 'STK4'
ksr_nw.loc[ksr_nw['source'] == 'PKD1', 'source'] = 'PRKD1'
ksr_nw.loc[ksr_nw['source'] == 'PKD2', 'source'] = 'PRKD2'
ksr_nw.loc[ksr_nw['source'] == 'PKD3', 'source'] = 'PRKD3'

# Map kinase gene name to UniProt ID and filter only those present in the human kinase dataset
ksr_nw['source_uid'] = ksr_nw['source'].map(pk_gname2uid)
ksr_nw = ksr_nw[ksr_nw['source_uid'].isin(human_pk['UniprotID'])]

ksr_lib = pd.read_parquet('../data/final_ksr_curated_kl90.parquet')
ksr_lib = ksr_lib[ksr_lib['source'].isin(human_pk['UniprotID'])]
ksr_lib = ksr_lib[ksr_lib['SOURCE'] != 'KinaseLibrary']

ksr_nw2 = ksr_nw[['source_uid', 'idx', 'mor']].copy()
ksr_nw2.columns = ['source', 'target', 'mor']
ksr_nw2['SOURCE'] = 'NetworKIN'
ksr_nw2 = ksr_nw2[~(ksr_nw2['source']+ksr_nw2['target']).isin(ksr_lib['source']+ksr_lib['target'])]

curatedlib = pd.concat([ksr_lib, ksr_nw2], ignore_index=True)
curatedlib.to_parquet(f'../results/{FILTER_TYPE}/ksr_curated_networkin.parquet')