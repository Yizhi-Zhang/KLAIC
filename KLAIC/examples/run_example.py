from pathlib import Path
import klaic

ROOT = Path(__file__).resolve().parents[1]
expr_path = ROOT / "examples" / "data" / "expr.tsv"

result = klaic.run_klaic(
    expr_path,
    library="KL95",
    context_filter="string400",
    n_workers=4,
)

result.activity.to_csv(ROOT / "examples" / "klaic_kinase_activity.csv")
result.ksr_library.to_csv(ROOT / "examples" / "klaic_ksr_library_with_context_flag.csv", index=False)
