\
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
STATS = ROOT / "artifacts" / "stats"
TABLES = ROOT / "artifacts" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def fmt_sci(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 1e3 or abs(x) < 1e-2:
        return f"{x:.2e}"
    return f"{x:.4f}"


def fmt_ci(lo: float, hi: float) -> str:
    if any([(v is None) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in [lo, hi]]):
        return "--"
    return f"[{fmt_sci(lo)}, {fmt_sci(hi)}]"


def write_table(tex: str, name: str):
    (TABLES / name).write_text(tex, encoding="utf-8")
    print(f"[tables] wrote {TABLES/name}")


def table_rosenbrock_best():
    f = STATS / "best_rosenbrock100d.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values("method")

    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Best LR & Mean loss & Std & 95\% CI & Div. rate \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['method']} & {r['lr']:.4g} & {fmt_sci(r['mean'])} & {fmt_sci(r['std'])} & {fmt_ci(r['ci95_low'], r['ci95_high'])} & {100*r['divergence_rate']:.0f}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), "table_rosenbrock100d_summary.tex")
    df.to_csv(TABLES / "table_rosenbrock100d_summary.csv", index=False)


def table_ablation():
    f = STATS / "summary_rosenbrock_ablation.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values("variant")

    # Display Diverged as text if 100%
    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Mean loss & Std & 95\% CI & Divergence \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        div = r["divergence_rate"]
        if div == 1.0:
            mean = "Diverged"
            std = "--"
            ci = "--"
        else:
            mean = fmt_sci(r["mean"])
            std = fmt_sci(r["std"])
            ci = fmt_ci(r["ci95_low"], r["ci95_high"])
        lines.append(f"{r['variant']} & {mean} & {std} & {ci} & {100*div:.0f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write_table("\n".join(lines), "table_rosenbrock_ablation.tex")


def table_pinn_suite():
    f = STATS / "summary_pinn_suite.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values(["problem", "pipeline"])

    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Problem & Pipeline & Mean final loss & Std & 95\% CI & Time (mean$\pm$std) \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        time_str = f"{r['time_mean']:.2f}$\\pm${r['time_std']:.2f}"
        lines.append(
            f"{r['problem']} & {r['pipeline']} & {fmt_sci(r['final_mean'])} & {fmt_sci(r['final_std'])} & {fmt_ci(r['final_ci95_low'], r['final_ci95_high'])} & {time_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write_table("\n".join(lines), "table_pinn_suite_summary.tex")


def table_pinn_warmup():
    f = STATS / "summary_pinn_suite_warmup.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values(["problem", "pipeline"])

    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Problem & Pipeline & Mean warmup loss & Std & 95\% CI \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['problem']} & {r['pipeline']} & {fmt_sci(r['warm_mean'])} & {fmt_sci(r['warm_std'])} & {fmt_ci(r['warm_ci95_low'], r['warm_ci95_high'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write_table("\n".join(lines), "table_pinn_suite_warmup.tex")


def table_quadratic():
    f = STATS / "summary_quadratic_sweep.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values(["condition_number", "method"])
    # simple longtable style
    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & $\kappa$ & Mean loss & Std & 95\% CI & Div. rate \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['method']} & {r['condition_number']:.0e} & {fmt_sci(r['mean'])} & {fmt_sci(r['std'])} & {fmt_ci(r['ci95_low'], r['ci95_high'])} & {100*r['divergence_rate']:.0f}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write_table("\n".join(lines), "table_quadratic_sweep_summary.tex")


def main():
    table_rosenbrock_best()
    table_ablation()
    table_pinn_suite()
    table_pinn_warmup()
    table_quadratic()


if __name__ == "__main__":
    main()
