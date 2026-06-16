import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict


# ---------- Z-score activity inference ----------
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
    

# ---------- Rank calculation ----------
def run_rank(act: pd.DataFrame, meta: pd.DataFrame, average: bool = True) -> pd.DataFrame:
    """
    Calculate the rank and scaled_rank for perturbed kinases in each experiment (after sign harmonization).

    Parameters
    ----------
    act : DataFrame
        Activity score matrix with kinases as rows and experiments as columns.
    meta : DataFrame
        Must contain at least:
        - id: experiment ID (corresponds to columns of act)
        - target: perturbed kinase(s); multiple kinases separated by ';'
        - sign: perturbation direction (+1 = activation/over-expression, -1 = inhibition/knockout)
    average : bool, default True
        If True, average the ranks (and scaled ranks) across experiments for the same kinase (as in R version).

    Returns
    -------
    DataFrame
        If average=True: columns are ['targets','rank','scaled_rank','sample']
        If average=False: columns are ['sample','targets','rank','kinases_act','all_kinases_act','scaled_rank']
    """
    # Prepare meta, remove experiments without targets
    obs = (
        meta.rename(columns={"target": "perturb"})
            .groupby("id", as_index=False)
            .agg(targets=("perturb", lambda x: ";".join(map(str, x))),
                 sign=("sign", lambda x: pd.unique(x)[0]))
            .rename(columns={"targets": "perturb", "id": "sample"})
    )
    obs = obs[(obs["perturb"].notna()) & (obs["perturb"] != "")]

    # Only keep experiments present in act
    samples = [c for c in act.columns if c in set(obs["sample"])]
    if not samples:
        return pd.DataFrame(columns=["sample","targets","rank","kinases_act",
                                     "all_kinases_act","scaled_rank"])

    # Wide format to long format
    method_act_long = (
        act.loc[:, samples]
           .reset_index()
           .rename(columns={"index": "kinase"})
           .melt(id_vars="kinase", var_name="sample", value_name="score")
           .dropna(subset=["score"])
    )

    rows = []
    for exp in method_act_long["sample"].unique():
        # Extract current experiment and sort by descending score
        sub = method_act_long[method_act_long["sample"] == exp].copy()
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        n_kin = len(sub)

        # Get targets of this experiment (may be multiple)
        if exp not in set(obs["sample"]):
            continue
        targets = (
            obs.loc[obs["sample"] == exp, "perturb"]
               .iloc[0].split(";")
        )

        # For each target, find its rank (1-based, as in R)
        for t in targets:
            idx = sub.index[sub["kinase"] == t]
            if len(idx) == 0:
                continue  # skip if not present (R sets NA then filters)
            rank = int(idx[0]) + 1
            rows.append({
                "sample": exp,
                "targets": t,
                "rank": rank,
                "kinases_act": n_kin,
                "all_kinases_act": ";".join(sub["kinase"].tolist()),
                "scaled_rank": rank / n_kin
            })

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df

    if average:
        # Aggregate ranks and scaled_ranks by target kinase; sample names combined with ';'
        agg = (rank_df
               .groupby("targets", as_index=False)
               .agg(rank=("rank", "mean"),
                    scaled_rank=("scaled_rank", "mean"),
                    sample=("sample", lambda x: ";".join(map(str, x)))))
        return agg

    return rank_df


# ---------- P_hit calculation ----------
def run_phit(act: pd.DataFrame, meta: pd.DataFrame, k: int = 10, average: bool = True, metric='rank') -> float:
    """
    Calculate P_hit(k): proportion of perturbed kinases that rank in the top-k.

    Parameters
    ----------
    act : DataFrame
        Kinase × experiment activity matrix.
    meta : DataFrame
        Contains columns id/target/sign.
    k : int
        Top-k threshold.
    average : bool
        Whether to average ranks across experiments for the same kinase.

    Returns
    -------
    float
        Hit rate (range 0–1).
    """
    rank_df = run_rank(act, meta, average=average)
    if rank_df.empty:
        return np.nan
    phit = np.mean(rank_df[metric] <= k)
    return float(phit)


# ---------- Recall@k calculation ----------
def compute_recall_at_k(meta: pd.DataFrame,
                        mat: pd.DataFrame,
                        k: int = 20,
                        mode: str = "macro",
                        metric='rank') -> dict:
    """
    Compute Recall@k = (number of true targets in top-k) / (total number of true targets)

    Parameters
    ----------
    meta : pd.DataFrame
        Should contain:
        - 'id': experiment ID (corresponds to columns of mat)
        - 'target': perturbed kinase(s) (possibly multiple separated by ';' or as multiple rows)
    mat : pd.DataFrame
        Kinase × experiment matrix, index as kinases, columns as experiments.
    k : int
        Top-k threshold.
    mode : {'macro', 'micro'}
        - 'macro': compute recall per experiment, then average across experiments (default)
        - 'micro': pool counts globally across all experiments

    Returns
    -------
    dict
        {
          'recall_at_k': float,
          'n_total_targets': int,
          'n_hits': int,
          'n_experiments': int,
          'recall_per_experiment': dict   # recall for each experiment
        }
    """
    recalls = {}
    n_hits_global = 0
    n_total_global = 0

    for exp in mat.columns:
        if exp not in meta["id"].values:
            continue

        # Get target set for this experiment
        targets = meta.loc[meta["id"] == exp, "target"].tolist()
        targets = [t for tg in targets for t in str(tg).split(";")]
        targets = [t for t in targets if t in mat.index]
        n_total = len(targets)
        if n_total == 0:
            continue

        # Get kinase ranked in top k
        pct = False if metric == 'rank' else 'True'
        ranks = mat[exp].rank(ascending=False, pct=pct)
        topk = set(ranks[ranks <= k].index)

        # Count hits
        hits = sum(t in topk for t in targets)
        recall = hits / n_total
        recalls[exp] = recall

        # Update global counts
        n_hits_global += hits
        n_total_global += n_total

    # Aggregate results
    if mode == "macro":
        recall_at_k = np.mean(list(recalls.values())) if recalls else np.nan
    elif mode == "micro":
        recall_at_k = n_hits_global / n_total_global if n_total_global else np.nan
    else:
        raise ValueError("mode must be 'macro' or 'micro'")

    return {
        "recall_at_k": recall_at_k,
        "n_total_targets": n_total_global,
        "n_hits": n_hits_global,
        "n_experiments": len(recalls),
        "recall_per_experiment": recalls
    }


# ---------- AUROC calculation ----------
## Utility: column scaling
def scale_scores(act: pd.DataFrame, scaling: str = "sd") -> pd.DataFrame:
    """
    Scale each experiment column by: 'sd' uses standard deviation, 'max' uses max(|.|).
    """
    act = act.copy()
    if scaling == "max":
        return act.apply(lambda s: s / np.nanmax(np.abs(s.values)) if np.nanmax(np.abs(s.values)) not in (0, np.nan) else s)
    elif scaling == "sd":
        # Equivalent to R::scale(center=FALSE, scale=TRUE): divide by column SD
        return act.apply(lambda s: s / np.nanstd(s.values) if np.nanstd(s.values) not in (0, np.nan) else s)
    else:
        return act

## Remove background experiments
def remove_bg(df_exp_rows: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """
    df_exp_rows: rows=experiment, columns=kinase; elements are "sign-harmonized scores" (may be NA)
    obs: index=experiment, columns: perturb (list[str])
    Logic: If all target kinase scores for this experiment sum to 0 or NA (i.e., no inferred activity), remove the experiment.
    """
    from typing import List
    
    keep_rows = []
    for exp, row in df_exp_rows.iterrows():
        targets: List[str] = obs.loc[exp, "perturb"]
        targets_in_mat = [t for t in targets if t in df_exp_rows.columns]
        if not targets_in_mat:
            # No target in matrix, equivalent to no information; remove
            continue
        s = row[targets_in_mat]
        sum_act = np.nansum(s.values)
        # Equivalent to R: !is.na(sum_act) & !sum_act == 0
        if not np.isnan(sum_act) and not (sum_act == 0):
            keep_rows.append(exp)
    return df_exp_rows.loc[keep_rows]

## Preprocessing: sign harmonization + reshaping
def prepare_bench(act: pd.DataFrame,
                  meta: pd.DataFrame,
                  rm_bg: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_exp_rows: rows=experiment, columns=kinase, elements are "sign-harmonized scores" (0->NA)
      obs: index=experiment, columns: perturb (list[str]) (sign is already absorbed into scores and thus set to 1)
    """
    
    # Prepare meta: merge targets per experiment, unique sign value per experiment
    target_df = (meta.rename(columns={"target": "perturb"})
                      .groupby("id", as_index=False)
                      .agg(targets=("perturb", lambda x: ";".join(map(str, x))),
                           sign=("sign", lambda x: pd.unique(x)[0]))
                      .rename(columns={"targets": "perturb", "id": "sample"}))
    # Remove empty targets
    target_df = target_df[(target_df["perturb"].notna()) & (target_df["perturb"] != "")]
    # Only keep experiments present in act
    target_df = target_df[target_df["sample"].isin(act.columns)]
    if target_df.empty:
        raise ValueError("No valid experiments remain after filtering.")

    # Sign harmonization: multiply each experiment column by sign
    # Build DataFrame: rows=experiment
    rows = []
    index_names = []
    for _, r in target_df.iterrows():
        exp = r["sample"]; sign = r["sign"]
        # Get the experiment column
        col = act[exp].copy()
        # Harmonize direction
        col = col * (1 if sign == 1 else -1)
        rows.append(col.values)
        index_names.append(exp)

    df_exp_rows = pd.DataFrame(rows, index=index_names, columns=act.index)

    # Convert 0 to NA (to match R logic)
    df_exp_rows = df_exp_rows.replace(0, np.nan)

    # obs: split target into list; set sign to 1 (already absorbed)
    obs = (target_df.assign(sign=1,
                            perturb=target_df["perturb"].apply(lambda s: s.split(";")))
                    .set_index("sample")[["perturb", "sign"]])

    # Optionally remove background experiments
    if rm_bg:
        df_exp_rows = remove_bg(df_exp_rows, obs)
        obs = obs.loc[df_exp_rows.index]

    return df_exp_rows, obs

## AUROC (rank-sum, no sklearn needed)
def _auroc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    AUROC = P(pos > neg); uses average rank for ties.
    """
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    pos_scores = pos_scores[~np.isnan(pos_scores)]
    neg_scores = neg_scores[~np.isnan(neg_scores)]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    x = np.concatenate([pos_scores, neg_scores])
    # Use pandas average rank
    ranks = pd.Series(x).rank(method="average").to_numpy()
    r_pos = ranks[:n_pos].sum()
    auroc = (r_pos - n_pos*(n_pos+1)/2.0) / (n_pos * n_neg)
    return float(auroc)

## Core: run benchmark and return AUROC (median)
def run_perturb_bench(act: pd.DataFrame,
                      meta: pd.DataFrame,
                      scale_data: bool = True,
                      rm_bg: bool = False,
                      n_iter: int = 1000,
                      metric: str = "auroc",
                      random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Equivalent to R's run_perturbBench (here, AUROC is implemented; to extend for AUPRC, add support as needed).
    Returns {'auroc': median_AUROC}
    """
    rng = np.random.default_rng(random_state)

    # 1) Optionally scale columns
    if scale_data:
        act = scale_scores(act, scaling="sd")

    # 2) Sign harmonization & reshape for benchmarking
    df_exp_rows, obs = prepare_bench(act, meta, rm_bg=rm_bg)

    # 3) Construct global positive/negative pools (across all experiments)
    pos_pool = []
    neg_pool = []
    for exp, row in df_exp_rows.iterrows():
        available_kin = row.index[row.notna()].tolist()
        if not available_kin:
            continue
        targets = [t for t in obs.loc[exp, "perturb"] if t in available_kin]
        if len(targets) == 0:
            continue
        # Positive class: perturbed kinases
        pos_pool.extend(row[targets].values.tolist())
        # Negative class: non-target kinases
        other_kin = [k for k in available_kin if k not in targets]
        neg_pool.extend(row[other_kin].values.tolist())

    pos_pool = np.array(pos_pool, dtype=float)
    neg_pool = np.array(neg_pool, dtype=float)
    pos_pool = pos_pool[~np.isnan(pos_pool)]
    neg_pool = neg_pool[~np.isnan(neg_pool)]

    if len(pos_pool) == 0 or len(neg_pool) == 0:
        return np.nan

    # 4) Downsample to balance, repeat AUROC calculation n_iter times
    n = min(len(pos_pool), len(neg_pool))
    aurocs = []
    for _ in range(n_iter):
        # Randomly sample equal numbers for positive and negative pools
        pos_idx = rng.choice(len(pos_pool), size=n, replace=(n > len(pos_pool)))
        neg_idx = rng.choice(len(neg_pool), size=n, replace=(n > len(neg_pool)))
        au = _auroc_from_scores(pos_pool[pos_idx], neg_pool[neg_idx])
        if not np.isnan(au):
            aurocs.append(au)

    if len(aurocs) == 0:
        return np.nan

    return aurocs

# ---------- Spatial label match calculation ----------
def add_spatial_label_match_only(
    ksr_df: pd.DataFrame,
    all_main_loc: list,
    source_label_col: str = "source_deeploc_label",
    target_label_col: str = "target_deeploc_label",
    label_sep: str = "|",
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
        x = x.replace("｜", "|")
        labels = {
            item.strip()
            for item in x.split(label_sep)
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