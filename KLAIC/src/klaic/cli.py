"""Command-line interface for KLAIC."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import SUPPORTED_FILTERS, SUPPORTED_LIBRARIES, run_klaic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KLAIC kinase activity inference.")
    parser.add_argument("--expr", required=True, help="Input phosphoproteomics log2FC matrix, CSV or TSV.")
    parser.add_argument("--library", default="KL95", choices=SUPPORTED_LIBRARIES)
    parser.add_argument("--filter", default="string400", choices=SUPPORTED_FILTERS, dest="context_filter")
    parser.add_argument("--outdir", default="klaic_output")
    parser.add_argument("--sep", default=None, help="Input separator. Default: infer from suffix.")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--minsize", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_klaic(
        args.expr,
        library=args.library,
        context_filter=args.context_filter,
        n_workers=args.n_workers,
        minsize=args.minsize,
        sep=args.sep,
        show_progress=not args.quiet,
    )

    if result.activity is not None:
        result.activity.to_csv(outdir / "klaic_kinase_activity.csv")
    result.ksr_library.to_csv(outdir / "klaic_ksr_library_with_context_flag.csv", index=False)

    print(f"KLAIC outputs written to: {outdir}")


if __name__ == "__main__":
    main()
