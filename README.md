# KLAIC: <ins>K</ins>inase <ins>L</ins>ibrary-based <ins>A</ins>ctivity <ins>I</ins>nference with <ins>C</ins>ontext constraints

![label1](https://img.shields.io/badge/license-MIT-green)

**| [Overview](#overview) | [Installation](#installation) | [Usage](#usage) | [Data](#data) | [Outputs](#outputs) | [Paper reproduction](#paper-reproduction) |**

---

## Overview

[Kinase Library (KL)](https://kinase-library.mit.edu/home) predictions greatly expand candidate kinase-substrate relationships (KSRs) but encode motif compatibility rather than cellular regulation. Using perturbation phosphoproteomes, we show that broad unfiltered expansion increases kinase coverage while compromising discrimination and directional accuracy. Subcellular co-localization and protein-association filtering improve the accuracy-coverage trade-off, providing practical guidance for context-constrained kinase activity inference.

**KLAIC** implements this framework as an end-to-end pipeline that converts a perturbation phosphoproteomics log2FC matrix into kinase activity scores. Given an input matrix, KLAIC:

1. **Scores** phosphosites with KL to derive candidate KSRs.
2. **Builds** a KSR library by integrating curated annotations from PhosphoSitePlus, PTMsigDB with iKiP-DB entries excluded, and the GPS 5.0 gold-standard set, together with Kinase Library predictions selected under user-defined confidence thresholds.
3. **Filters** KL-derived KSRs by cellular context — subcellular co-localization from [DeepLoc](https://services.healthtech.dtu.dk/services/DeepLoc-2.1/) or protein-protein association from [STRING](https://string-db.org/) and [Buljan et al.](https://doi.org/10.1016/j.molcel.2020.07.001) — to retain biologically plausible interactions.
4. **Infers** kinase activity using RoKAI-style z-score aggregation over context-approved substrates.

---

## Installation

### Prerequisites

- Python >= 3.10
- pip

### Install from source

```bash
git clone https://github.com/ZhangMenghuan-Tongji/KLAIC.git
cd KLAIC/KLAIC
pip install .
```

Key dependencies are installed automatically.

---

## Usage

### 1. Python API

The simplest way to run the full pipeline:

```python
import klaic

result = klaic.run_klaic(
    "expr.tsv",
    library="KL95",
    context_filter="string400",
    n_workers=16,
    minsize=5,
)

activity = result.activity          # kinase activity matrix
ksr_library = result.ksr_library    # selected KSR library with context_flag
```

`run_klaic()` prints step-by-step progress to the console. Set `show_progress=False` to suppress both the progress bar and log messages.

You can also run individual steps:

```python
import klaic

# Step 1: KL scoring
kl_ksr, kl_errors = klaic.run_kinase_library("expr.tsv", n_workers=16)

# Step 2: Build KSR library
ksr_lib = klaic.generate_ksr_library("expr.tsv", kl_ksr=kl_ksr, library="KL95")

# Step 3: Apply context filter
ksr_lib = klaic.apply_context_filter(ksr_lib, context_filter="string400")

# Step 4: Infer kinase activity
activity = klaic.infer_activity("expr.tsv", ksr_lib, minsize=5)
```

### 2. Command line

```bash
klaic --expr expr.tsv --library KL95 --filter string400 --outdir results --n-workers 16
```

For a full list of arguments:

```bash
klaic --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--expr` | *(required)* | Input phosphoproteomics log2FC matrix (CSV or TSV) |
| `--library` | `KL95` | KSR library preset (see [Supported options](#supported-options)) |
| `--filter` | `string400` | Context filter (see [Supported options](#supported-options)) |
| `--outdir` | `klaic_output` | Output directory |
| `--n-workers` | `16` | Parallel workers for KL scoring |
| `--minsize` | `5` | Minimum quantified substrates per kinase for z-score inference |
| `--sep` | auto | Column separator; inferred from file suffix if omitted |
| `--quiet` | off | Disable progress bar and log messages |

### 3. Run the bundled example

```bash
cd KLAIC
pip install .
python examples/run_example.py
```

Expected outputs are written to `examples/`:

```
KLAIC/examples/
├── data/
│   └── expr.tsv                              # example input matrix
├── klaic_kinase_activity.csv                 # kinase activity scores
└── klaic_ksr_library_with_context_flag.csv   # KSR library with context_flag
```

---

## Data

### Input matrix

KLAIC expects a **CSV or TSV file** with:

- **Rows:** phosphosite IDs in the form `UniProtID_residuePosition`, e.g. `O60343_S318`
- **Columns:** samples / perturbation conditions
- **Values:** perturbation log2 fold-change (log2FC)

Missing values are allowed and handled during z-score inference.

### Supported options

**KSR libraries** (`library` / `--library`):

| Option | Description |
|--------|-------------|
| `Curated` | Curated KSRs only (KL step skipped) |
| `KL90` | Curated + KL-derived KSRs with mor > 0.90 |
| `KL95` | Curated + KL-derived KSRs with mor > 0.95 *(default)* |
| `KL97.5` | Curated + KL-derived KSRs with mor > 0.975 |
| `KL99` | Curated + KL-derived KSRs with mor > 0.99 |
| `KL99-T10` | KL99 + top 10 KL-derived KSRs per kinase |
| `KL99-T20` | KL99 + top 20 KL-derived KSRs per kinase |
| `KL99-T50` | KL99 + top 50 KL-derived KSRs per kinase |

**Context filters** (`context_filter` / `--filter`):

| Option | Description |
|--------|-------------|
| `unfiltered` | No context filtering; all KSRs retained |
| `string700` | Retain KSRs where kinase and substrate proteins are STRING v12.0 partners with a combined score ≥ 700, or are supported by kinase–protein associations from Buljan et al. |
| `string400` | Retain KSRs where kinase and substrate proteins are STRING v12.0 partners with a combined score ≥ 400, or are supported by kinase–protein associations from Buljan et al. *(default)* |
| `deeploc` | Retain KSRs where kinase and substrate share at least one DeepLoc main subcellular compartment |

> Curated KSRs are always assigned `context_flag = True` regardless of the filter.

### Built-in reference data

The following files are bundled under `src/klaic/data/` and loaded automatically at runtime:

```
src/klaic/data/
├── curated_lib.csv                    # curated KSRs from 
├── deep_loc.pkl                       # DeepLoc subcellular localizations
├── kl_kinase_gname2uid.pkl            # KL-kinase gene name → UniProt ID map
├── string400.pkl                      # STRING protein pairs with combined score ≥ 400, supplemented by kinase–protein associations from Buljan et al.
├── string700.pkl                      # STRING protein pairs with combined score ≥ 700, supplemented by kinase–protein associations from Buljan et al.
└── uniprot_human_protein_fasta.pkl    # human protein sequences
```

---

## Outputs

### `run_klaic()` return value

`run_klaic()` returns a `KLAICResult` object with the following fields:

| Field | Description |
|-------|-------------|
| `activity` | Kinase activity matrix; rows are kinase UniProt IDs, columns are input samples |
| `ksr_library` | Selected KSR library with `context_flag` column |
| `kl_ksr` | Raw KL-derived KSRs before curated de-duplication (`None` if `library="Curated"`) |
| `kl_errors` | Per-site KL errors, if any |

### CLI outputs

Running the `klaic` command writes the following files to `--outdir`:

```
results/
├── klaic_kinase_activity.csv                 # kinase activity matrix
└── klaic_ksr_library_with_context_flag.csv # KSR library with context_flag
```

---

## Paper reproduction

Scripts and pre-computed results for reproducing the benchmark analyses reported in the paper are provided in the sibling `paper_code/` directory at the repository root:

```
paper_code/
├── data/           # benchmark phosphoproteomics matrix and metadata
├── scripts/        # scripts for KSR library construction, kinase activity inference, and benchmark evaluation
└── results/        # pre-computed benchmark tables, generated KSR libraries, and kinase activity results
```

To reproduce the full benchmark from scratch:

```bash
cd paper_code/scripts

# Step 1: Build KSR libraries under each context filter
python 1-1.process_unfiltered_library.py
python 1-2.process_string_library.py
python 1-3.process_deeploc_library.py

# Step 2: Infer kinase activity for all library × filter combinations
python 2.infer_activity.py

# Step 3: Run benchmark evaluation and merge results
python 3-1.run_evaluation.py
python 3-2.merge_evaluation_results.py
```

Pre-computed benchmark summaries are available in `paper_code/results/benchmark_results.csv` and `paper_code/results/benchmark_results_same_kinase.csv`.
