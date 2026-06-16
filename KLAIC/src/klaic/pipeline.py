"""High-level KLAIC pipeline wrappers around the notebook-preserved functions."""

from __future__ import annotations

import pickle
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from . import functions
from .functions import add_spatial_label_match_only, run_zscore

SUPPORTED_LIBRARIES = (
    "Curated",
    "KL99-T10",
    "KL99-T20",
    "KL99-T50",
    "KL99",
    "KL97.5",
    "KL95",
    "KL90",
)

SUPPORTED_FILTERS = (
    "unfiltered",
    "string700",
    "string400",
    "deeploc",
)

DEEPLOC_MAIN_LOCS = [
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell membrane",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


class KLAICResult(dict):
    """Dictionary-like result with convenient attributes.

    Keys are ``activity``, ``ksr_library``, ``kl_ksr``, and ``kl_errors``.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _resource_path(filename: str) -> Path:
    path = resources.files("klaic").joinpath("data", filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing built-in KLAIC data file: {filename}. "
            "Put it under src/klaic/data/ before installing/building the package."
        )
    return Path(str(path))


def _read_pickle_resource(filename: str):
    with _resource_path(filename).open("rb") as f:
        return pickle.load(f)


def _read_csv_resource(filename: str) -> pd.DataFrame:
    return pd.read_csv(_resource_path(filename))


def load_expr(expr: str | Path | pd.DataFrame, sep: str | None = None) -> pd.DataFrame:
    """Load a KLAIC input expression matrix.

    The matrix must have phosphosite IDs such as ``O60343_S318`` as rows and
    samples/conditions as columns. Values should be perturbation log2FC.
    """
    if isinstance(expr, pd.DataFrame):
        mat = expr.copy()
    else:
        path = Path(expr)
        if sep is None:
            if path.suffix.lower() in {".tsv", ".txt"}:
                sep = "\t"
            else:
                sep = ","
        mat = pd.read_csv(path, sep=sep, index_col=0)

    mat.index = mat.index.astype(str)
    return mat.apply(pd.to_numeric, errors="coerce")


def _make_site_df(expr: pd.DataFrame) -> pd.DataFrame:
    site_df = pd.DataFrame(expr.index, columns=['idx'])
    site_df['UID'] = site_df['idx'].str.split('_').str[0]
    site_df['MOD_RSD'] = site_df['idx'].str.split('_').str[1]
    site_df['RSD'] = site_df['MOD_RSD'].str.extract(r'([A-Z])')
    site_df['Pos'] = site_df['MOD_RSD'].str.extract(r'(\d+)').astype(int)
    site_df = site_df[site_df['RSD'].isin(['S', 'T', 'Y'])]
    return site_df


def _init_worker(protein_fasta: dict, kl_threshold: float):
    functions.ufa = protein_fasta
    functions.KL_THRESHOLD = kl_threshold


def run_kinase_library(
    expr: str | Path | pd.DataFrame,
    *,
    kl_threshold: float = 90,
    n_workers: int = 16,
    sep: str | None = None,
    show_progress: bool = True,
    verbose: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Kinase Library percentile scoring for all valid S/T/Y sites.

    Returns ``(kl_ksr, error_df)``. ``kl_ksr`` contains columns ``source``,
    ``mor``, ``target``, and ``SOURCE``.
    """
    if verbose is None:
        verbose = show_progress

    mat = load_expr(expr, sep=sep)
    site_df = _make_site_df(mat)

    _log(
        f"[KLAIC] Kinase Library: scoring {len(site_df)} valid S/T/Y sites "
        f"(threshold={kl_threshold}, workers={n_workers})...",
        verbose=verbose,
    )

    protein_fasta = _read_pickle_resource("uniprot_human_protein_fasta.pkl")
    kl_kinase_gname2uid = _read_pickle_resource("kl_kinase_gname2uid.pkl")

    functions.ufa = protein_fasta
    functions.KL_THRESHOLD = kl_threshold

    tasks = list(
        site_df[['UID', 'Pos', 'idx']]
        .itertuples(index=False, name=None)
    )

    iterator = tqdm(tasks, total=len(tasks), disable=not show_progress)

    if n_workers == 1:
        results = [functions.process_one_site(uid, pos, site_idx) for uid, pos, site_idx in iterator]
    else:
        results = Parallel(
            n_jobs=n_workers,
            backend="loky",
            batch_size=50,
            verbose=0,
            initializer=_init_worker,
            initargs=(protein_fasta, kl_threshold),
        )(
            delayed(functions.process_one_site)(uid, pos, site_idx)
            for uid, pos, site_idx in iterator
        )

    ksr_list = [
        r for r in results
        if isinstance(r, pd.DataFrame) and not r.empty
    ]

    error_list = [
        r for r in results
        if isinstance(r, dict)
    ]

    if len(ksr_list) > 0:
        kl_ksr = pd.concat(ksr_list, ignore_index=True)
    else:
        kl_ksr = pd.DataFrame(columns=['source', 'mor', 'target'])

    error_df = pd.DataFrame(error_list)

    kl_ksr['source'] = kl_ksr['source'].map(kl_kinase_gname2uid)
    kl_ksr['mor'] = kl_ksr['mor'] / 100
    kl_ksr['SOURCE'] = 'KinaseLibrary'

    _log(
        f"[KLAIC] Kinase Library done: {len(kl_ksr)} KSR entries, "
        f"{len(error_df)} site-level errors.",
        verbose=verbose,
    )

    return kl_ksr, error_df


def generate_ksr_library(
    expr: str | Path | pd.DataFrame,
    kl_ksr: pd.DataFrame | None = None,
    *,
    library: str = "KL95",
    sep: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the requested curated/KL KSR library before context filtering."""
    if library not in SUPPORTED_LIBRARIES:
        raise ValueError(f"Invalid library: {library}. Choose from {SUPPORTED_LIBRARIES}.")

    _log(f"[KLAIC] Generating KSR library: {library}...", verbose=verbose)

    mat = load_expr(expr, sep=sep)
    curated_lib = _read_csv_resource("curated_lib.csv")

    if library == "Curated":
        merged_ksr = curated_lib.copy()
    else:
        if kl_ksr is None:
            kl_ksr, _ = run_kinase_library(mat, kl_threshold=90, verbose=verbose)

        kl_ksr_filtered = kl_ksr.merge(
            curated_lib[['source', 'target']],
            on=['source', 'target'],
            how='left',
            indicator=True
        )
        kl_ksr_not_in_curated = kl_ksr_filtered[kl_ksr_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])
        merged_ksr = pd.concat([curated_lib, kl_ksr_not_in_curated], ignore_index=True)

    merged_ksr = merged_ksr[merged_ksr['target'].isin(mat.index)]

    if library == 'KL90':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.9].copy()
    elif library == 'KL95':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.95].copy()
    elif library == 'KL97.5':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.975].copy()
    elif library == 'KL99':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.99].copy()
    elif library == 'KL99-T50':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.99].copy()

        def keep_top_kl(df):
            kl = df[df['SOURCE'] == 'KinaseLibrary'].sort_values('mor', ascending=False).head(50)
            return pd.concat([df[df['SOURCE'] != 'KinaseLibrary'], kl])

        ksr_lib_filtered = (
            ksr_lib.groupby('source', group_keys=False)
            .apply(keep_top_kl)
            .reset_index(drop=True)
        )
        ksr_lib = ksr_lib_filtered
    elif library == 'KL99-T20':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.99].copy()

        def keep_top_kl(df):
            kl = df[df['SOURCE'] == 'KinaseLibrary'].sort_values('mor', ascending=False).head(20)
            return pd.concat([df[df['SOURCE'] != 'KinaseLibrary'], kl])

        ksr_lib_filtered = (
            ksr_lib.groupby('source', group_keys=False)
            .apply(keep_top_kl)
            .reset_index(drop=True)
        )
        ksr_lib = ksr_lib_filtered
    elif library == 'KL99-T10':
        ksr_lib = merged_ksr[merged_ksr['mor'] > 0.99].copy()

        def keep_top_kl(df):
            kl = df[df['SOURCE'] == 'KinaseLibrary'].sort_values('mor', ascending=False).head(10)
            return pd.concat([df[df['SOURCE'] != 'KinaseLibrary'], kl])

        ksr_lib_filtered = (
            ksr_lib.groupby('source', group_keys=False)
            .apply(keep_top_kl)
            .reset_index(drop=True)
        )
        ksr_lib = ksr_lib_filtered
    elif library == 'Curated':
        ksr_lib = merged_ksr[merged_ksr['SOURCE'] != 'KinaseLibrary'].copy()
    else:
        raise ValueError(f"Invalid library: {library}")

    ksr_lib['source_uid'] = ksr_lib['source'].tolist()
    ksr_lib['target_uid'] = ksr_lib['target'].str.split('_').str[0].tolist()

    n_kinases = ksr_lib['source'].nunique()
    n_targets = ksr_lib['target'].nunique()
    _log(
        f"[KLAIC] KSR library ready: {len(ksr_lib)} relationships "
        f"({n_kinases} kinases, {n_targets} phosphosites).",
        verbose=verbose,
    )

    return ksr_lib


def apply_context_filter(
    ksr_lib: pd.DataFrame,
    *,
    context_filter: str = "string400",
    verbose: bool = True,
) -> pd.DataFrame:
    """Add ``context_flag`` according to the requested context filter."""
    if context_filter not in SUPPORTED_FILTERS:
        raise ValueError(f"Invalid filter: {context_filter}. Choose from {SUPPORTED_FILTERS}.")

    _log(f"[KLAIC] Applying context filter: {context_filter}...", verbose=verbose)

    ksr_lib = ksr_lib.copy()

    if context_filter == 'unfiltered':
        ksr_lib['context_flag'] = True

    elif context_filter == 'string400':
        string_prot_pairs = _read_pickle_resource("string400.pkl")

        def check_string_co_func(row):
            # If SOURCE is Curated or PhosphoSitePlus, return True directly
            if row['SOURCE'] == 'Curated':
                return True
            # Check if target_protein is missing
            if pd.isnull(row['target_uid']):
                return False
            return tuple(sorted((row['source'], row['target_uid']))) in string_prot_pairs
        ksr_lib['context_flag'] = ksr_lib.apply(check_string_co_func, axis=1)

    elif context_filter == 'string700':
        string_prot_pairs = _read_pickle_resource("string700.pkl")

        def check_string_co_func(row):
            # If SOURCE is Curated or PhosphoSitePlus, return True directly
            if row['SOURCE'] == 'Curated':
                return True
            # Check if target_protein is missing
            if pd.isnull(row['target_uid']):
                return False
            return tuple(sorted((row['source'], row['target_uid']))) in string_prot_pairs
        ksr_lib['context_flag'] = ksr_lib.apply(check_string_co_func, axis=1)

    elif context_filter == 'deeploc':
        deep_loc = _read_pickle_resource("deep_loc.pkl")

        ksr_lib['source_Localizations'] = ksr_lib['source_uid'].map(deep_loc)
        ksr_lib['target_Localizations'] = ksr_lib['target_uid'].map(deep_loc)

        ksr_lib = add_spatial_label_match_only(
            ksr_lib,
            all_main_loc=DEEPLOC_MAIN_LOCS,
            source_label_col="source_Localizations",
            target_label_col="target_Localizations",
            source_type_col="SOURCE",
            curated_values={"Curated"},
            label_match_col="context_flag",
        )

    else:
        raise ValueError(f"Invalid filter: {context_filter}")

    n_passed = int(ksr_lib['context_flag'].sum())
    if len(ksr_lib) > 0:
        pct = 100 * n_passed / len(ksr_lib)
        _log(
            f"[KLAIC] Context filter done: {n_passed}/{len(ksr_lib)} KSRs passed "
            f"({pct:.1f}%).",
            verbose=verbose,
        )
    else:
        _log("[KLAIC] Context filter done: empty KSR library.", verbose=verbose)

    return ksr_lib


def infer_activity(
    expr: str | Path | pd.DataFrame,
    ksr_lib: pd.DataFrame,
    *,
    minsize: int = 5,
    sep: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Infer kinase activity using rows with ``context_flag == True`` and ``mor > 0``."""
    _log(f"[KLAIC] Inferring kinase activity (minsize={minsize})...", verbose=verbose)

    mat = load_expr(expr, sep=sep)
    ksr_lib_used = ksr_lib.copy()
    ksr_lib_used = ksr_lib_used[ksr_lib_used['context_flag']==True]
    ksr_lib_used = ksr_lib_used[ksr_lib_used['mor'] > 0]

    _log(
        f"[KLAIC] Using {len(ksr_lib_used)} KSRs from "
        f"{ksr_lib_used['source'].nunique()} kinases for z-score inference.",
        verbose=verbose,
    )

    activity = run_zscore(mat, ksr_lib_used, minsize=minsize)

    if activity is None:
        _log("[KLAIC] Activity inference returned no result.", verbose=verbose)
    else:
        _log(
            f"[KLAIC] Activity inference done: {activity.shape[0]} kinases x "
            f"{activity.shape[1]} samples.",
            verbose=verbose,
        )

    return activity


def run_klaic(
    expr: str | Path | pd.DataFrame,
    *,
    library: str = "KL95",
    context_filter: str = "string400",
    n_workers: int = 16,
    minsize: int = 5,
    sep: str | None = None,
    show_progress: bool = True,
) -> KLAICResult:
    """Run the full KLAIC pipeline.

    Parameters
    ----------
    expr:
        DataFrame or path to a CSV/TSV expression matrix. Rows are phosphosites
        such as ``O60343_S318``. Columns are samples. Values are log2FC.
    library:
        One of ``SUPPORTED_LIBRARIES``.
    context_filter:
        One of ``SUPPORTED_FILTERS``.
    n_workers:
        Parallel workers for Kinase Library prediction.
    minsize:
        Minimum quantified substrates per kinase for z-score inference.

    Returns
    -------
    KLAICResult
        ``result.activity`` is the kinase activity matrix. ``result.ksr_library``
        is the selected KSR library with ``context_flag``.
    """
    verbose = show_progress
    mat = load_expr(expr, sep=sep)

    _log(
        f"[KLAIC] Pipeline started: library={library}, "
        f"context_filter={context_filter}, minsize={minsize}.",
        verbose=verbose,
    )
    _log(
        f"[KLAIC] Input matrix: {mat.shape[0]} phosphosites x "
        f"{mat.shape[1]} samples.",
        verbose=verbose,
    )

    kl_ksr = None
    kl_errors = pd.DataFrame()
    if library != "Curated":
        _log("[KLAIC] Step 1/4: Kinase Library scoring...", verbose=verbose)
        kl_ksr, kl_errors = run_kinase_library(
            mat,
            kl_threshold=90,
            n_workers=n_workers,
            show_progress=show_progress,
            verbose=verbose,
        )
    else:
        _log(
            "[KLAIC] Step 1/4: Kinase Library scoring skipped (Curated library).",
            verbose=verbose,
        )

    _log("[KLAIC] Step 2/4: Generating KSR library...", verbose=verbose)
    ksr_lib = generate_ksr_library(
        mat, kl_ksr=kl_ksr, library=library, verbose=verbose,
    )

    _log("[KLAIC] Step 3/4: Applying context filter...", verbose=verbose)
    ksr_lib = apply_context_filter(
        ksr_lib, context_filter=context_filter, verbose=verbose,
    )

    _log("[KLAIC] Step 4/4: Inferring kinase activity...", verbose=verbose)
    activity = infer_activity(mat, ksr_lib, minsize=minsize, verbose=verbose)

    _log("[KLAIC] Pipeline complete.", verbose=verbose)

    return KLAICResult(
        activity=activity,
        ksr_library=ksr_lib,
        kl_ksr=kl_ksr,
        kl_errors=kl_errors,
    )
