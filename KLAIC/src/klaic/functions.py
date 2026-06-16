"""Notebook-defined functions preserved for KLAIC.

The function bodies in this module are copied from the provided notebook and should
not be edited unless the notebook definition itself is intentionally updated.
"""

import numpy as np
import pandas as pd
from typing import Optional

import kinase_library as kl


# Notebook globals used by process_one_site. Pipeline wrappers set these before use.
ufa = {}
KL_THRESHOLD = 90


def extract_flank_for_kl(seq, pos, flank=7):
    """
    从蛋白序列中提取上下游氨基酸，1-index。
    中心位点小写并加星号，不足补'_'
    """
    seq = seq.strip()
    L = len(seq)
    i = pos - 1  # 转为0-index
    left = max(0, i - flank)
    right = min(L, i + flank + 1)
    window = seq[left:right]
    
    # 补齐长度
    left_pad = '_' * (flank - (i - left))
    right_pad = '_' * (flank - (right - i - 1))
    window = left_pad + window + right_pad
    
    # 替换中心位点
    center_idx = flank
    window = window[:center_idx].upper() + window[center_idx].lower() + window[center_idx+1:].upper()
    
    return window


def process_one_site(uid, pos, site_idx):
    """
    单个位点计算 KL percentile，并返回该位点对应的 KSR dataframe。
    出错时返回错误信息，方便后面检查。
    """
    try:
        if uid not in ufa or not isinstance(ufa[uid], str):
            return None

        seq = ufa[uid]
        peptide = extract_flank_for_kl(seq, int(pos), flank=7)

        s = kl.Substrate(peptide)
        tmp = pd.DataFrame(s.percentile()).reset_index()
        tmp.columns = ['source', 'mor']

        tmp = tmp[tmp['mor'] > KL_THRESHOLD].copy()

        if tmp.empty:
            return None

        tmp['target'] = site_idx
        return tmp

    except Exception as e:
        return {
            "idx": site_idx,
            "uid": uid,
            "pos": pos,
            "error": repr(e)
        }


def add_spatial_label_match_only(
    ksr_df: pd.DataFrame,
    all_main_loc: list,
    source_label_col: str = "source_deeploc_label",
    target_label_col: str = "target_deeploc_label",
    source_type_col: str | None = None,
    curated_values: set | list | tuple | None = None,
    label_match_col: str = "deeploc_label_match",
) -> pd.DataFrame:
    """
    Only retain label_match-related calculation: Determine whether there is an intersection between the DeepLoc main localization labels of source/target for each row.
    If the row is curated (determined by source_type_col and curated_values), set as True directly.
    Finally, append the label_match column to the end of ksr_df.
    """

    df = ksr_df.copy()

    # Make sure the columns exist
    if source_label_col not in df.columns:
        raise ValueError(f"Missing source label column: {source_label_col}")
    if target_label_col not in df.columns:
        raise ValueError(f"Missing target label column: {target_label_col}")

    # Index dictionary for main_loc
    loc_to_idx = {loc: i for i, loc in enumerate(all_main_loc)}

    # Multilabel parsing function
    missing_tokens = {"", "nan", "none", "na", "n/a", "unknown", "null"}
    def parse_label_set(x):
        if pd.isna(x):
            return set()
        x = str(x).strip()
        if x.lower() in missing_tokens:
            return set()
        labels = {
            item.strip()
            for item in x.split('|')
            if item.strip() and item.strip().lower() not in missing_tokens
        }
        labels = {loc for loc in labels if loc in loc_to_idx}
        return labels

    source_label_sets = df[source_label_col].apply(parse_label_set).tolist()
    target_label_sets = df[target_label_col].apply(parse_label_set).tolist()

    # Mark curated rows
    if source_type_col is not None and curated_values is not None:
        if source_type_col not in df.columns:
            raise ValueError(f"Missing source type column: {source_type_col}")
        curated_values = set(curated_values)
        curated_mask = df[source_type_col].isin(curated_values).to_numpy()
    else:
        curated_mask = np.zeros(len(df), dtype=bool)

    n = len(df)
    label_match = np.full(n, False, dtype=bool)

    for i in range(n):
        # If curated, set as True directly
        if curated_mask[i]:
            label_match[i] = True
            continue

        s_labels = source_label_sets[i]
        t_labels = target_label_sets[i]
        shared_labels = s_labels & t_labels
        if len(shared_labels) > 0:
            label_match[i] = True
        else:
            label_match[i] = False

    # Append the label_match column at the end of ksr_df (does not affect the original order, only adds a new column at the end)
    df[label_match_col] = label_match
    return df


def run_zscore(mat: pd.DataFrame,
               network: pd.DataFrame,
               minsize: int = 5) -> Optional[pd.DataFrame]:
    """
    Z-score enrichment scoring (equivalent to the given R version):
    For each experimental column c, take the non-missing values V = mat[c], filter the network to keep only edges whose target is in index(V),
    and exclude sources with fewer than 'minsize' targets. Construct the (source × target) weight matrix W=mor.
    Let S = sd(V). The score for each source is calculated as:
        score = (W @ V) / ( S * sqrt( |W| @ 1 ) )
    where @ is matrix multiplication and 1 is an all-one vector of length |targets|.
    Finally, concatenate the scores for all experiments into a (source × condition) wide DataFrame and return.
    """    
    scores_rows = []

    # For each experimental column (condition)
    for cond in mat.columns:
        V = mat[cond].dropna()
        if V.empty:
            continue

        # Filter network to keep only edges whose target is in V.index
        nf = network[network["target"].isin(V.index)].copy()
        if nf.empty:
            continue

        # Exclude sources with fewer than 'minsize' targets
        nf["_n"] = nf.groupby("source")["target"].transform("size")
        nf = nf[nf["_n"] >= minsize]
        if nf.empty:
            continue

        # Construct source × target matrix for 'mor' (fill missing with 0)
        kin_sub = nf.pivot_table(index="source", columns="target",
                                 values="mor", aggfunc="first", fill_value=0.0)

        # Determine common targets with V, and align order
        valid_targets = kin_sub.columns.intersection(V.index)
        if len(valid_targets) == 0:
            continue
        A = kin_sub.loc[:, valid_targets].to_numpy(dtype=float)
        v = V.loc[valid_targets].to_numpy(dtype=float).reshape(-1, 1)

        # Sample standard deviation (using ddof=1, compatible with R)
        S = float(np.std(v, ddof=1))
        if np.isnan(S) or S == 0.0:
            continue

        # Z-score calculation
        num = A @ v                                            # shape: (n_source, 1)
        den = S * np.sqrt(np.abs(A) @ np.ones((A.shape[1], 1)))  # shape: (n_source, 1)
        z = (num / den).ravel()

        # Construct result DataFrame for this condition
        df_cond = pd.DataFrame({
            "source": kin_sub.index,
            "condition": cond,
            "score": z
        })
        # Remove NaN
        df_cond = df_cond[np.isfinite(df_cond["score"])]
        if not df_cond.empty:
            scores_rows.append(df_cond)

    if not scores_rows:
        return None

    scores = pd.concat(scores_rows, ignore_index=True)

    # Convert to wide format: rows=source, columns=condition, values=score
    act = scores.pivot_table(index="source", columns="condition",
                             values="score", aggfunc="first")
    act.index.name = None
    act.columns.name = None
    
    # Return plain DataFrame (not MultiIndex)
    return act
