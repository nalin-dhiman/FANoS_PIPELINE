\
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "artifacts" / "plots"
TABLES = ROOT / "artifacts" / "tables"
OUT = ROOT / "latex_snippets"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Figures snippet
    fig_lines = []
    fig_lines.append("% Auto-generated: include figures produced by the results pipeline.")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.85\\linewidth]{{{(PLOTS/'fig_rosenbrock100d_lr_sweep.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{Rosenbrock-100D learning-rate sweep (mean final loss; 95\\% bootstrap CI; 10 seeds; 3000 gradient evaluations).}")
    fig_lines.append("  \\label{fig:rosenbrock_sweep}")
    fig_lines.append("\\end{figure}")
    fig_lines.append("")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.85\\linewidth]{{{(PLOTS/'fig_rosenbrock_ablation.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{Rosenbrock-100D ablations (mean final loss; 95\\% bootstrap CI; 5 seeds).}")
    fig_lines.append("  \\label{fig:ablation}")
    fig_lines.append("\\end{figure}")
    fig_lines.append("")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.95\\linewidth]{{{(PLOTS/'fig_pinn_suite_final_loss.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{PINN warm-start suite: distribution of final residual loss after L-BFGS refinement (5 seeds).}")
    fig_lines.append("  \\label{fig:pinn_final}")
    fig_lines.append("\\end{figure}")
    fig_lines.append("")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.95\\linewidth]{{{(PLOTS/'fig_pinn_suite_warmup_loss.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{PINN warm-start suite: warm-start loss before L-BFGS (5 seeds).}")
    fig_lines.append("  \\label{fig:pinn_warmup}")
    fig_lines.append("\\end{figure}")
    fig_lines.append("")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.85\\linewidth]{{{(PLOTS/'fig_quadratic_sweep.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{Ill-conditioned quadratic diagnostic sweep (mean final loss; 95\\% bootstrap CI).}")
    fig_lines.append("  \\label{fig:quadratic}")
    fig_lines.append("\\end{figure}")
    fig_lines.append("")
    fig_lines.append("\\begin{figure}[t]")
    fig_lines.append("  \\centering")
    fig_lines.append(f"  \\includegraphics[width=0.85\\linewidth]{{{(PLOTS/'fig_thermostat_diagnostics.pdf').as_posix()}}}")
    fig_lines.append("  \\caption{Thermostat diagnostics: friction coefficient $\\zeta$ over time for two Rosenbrock regimes.}")
    fig_lines.append("  \\label{fig:thermostat}")
    fig_lines.append("\\end{figure}")

    (OUT / "INCLUDE_FIGURES.tex").write_text("\n".join(fig_lines), encoding="utf-8")

    tab_lines = []
    tab_lines.append("% Auto-generated: include tables produced by the results pipeline.")
    tab_lines.append("\\begin{table}[t]")
    tab_lines.append("  \\centering")
    tab_lines.append(f"  \\input{{{(TABLES/'table_rosenbrock100d_summary.tex').as_posix()}}}")
    tab_lines.append("  \\caption{Rosenbrock-100D: best learning rate per method (mean final loss; 95\\% bootstrap CI; 10 seeds).}")
    tab_lines.append("  \\label{tab:rosenbrock_summary}")
    tab_lines.append("\\end{table}")
    tab_lines.append("")
    tab_lines.append("\\begin{table}[t]")
    tab_lines.append("  \\centering")
    tab_lines.append(f"  \\input{{{(TABLES/'table_rosenbrock_ablation.tex').as_posix()}}}")
    tab_lines.append("  \\caption{Rosenbrock-100D: ablation study (5 seeds).}")
    tab_lines.append("  \\label{tab:ablation}")
    tab_lines.append("\\end{table}")
    tab_lines.append("")
    tab_lines.append("\\begin{table}[t]")
    tab_lines.append("  \\centering")
    tab_lines.append(f"  \\input{{{(TABLES/'table_pinn_suite_summary.tex').as_posix()}}}")
    tab_lines.append("  \\caption{PINN warm-start suite: mean final loss after L-BFGS refinement and runtime (5 seeds).}")
    tab_lines.append("  \\label{tab:pinn_suite}")
    tab_lines.append("\\end{table}")
    tab_lines.append("")
    tab_lines.append("\\begin{table}[t]")
    tab_lines.append("  \\centering")
    tab_lines.append(f"  \\input{{{(TABLES/'table_pinn_suite_warmup.tex').as_posix()}}}")
    tab_lines.append("  \\caption{PINN warm-start suite: warmup losses before L-BFGS (5 seeds).}")
    tab_lines.append("  \\label{tab:pinn_warmup}")
    tab_lines.append("\\end{table}")
    tab_lines.append("")
    tab_lines.append("\\begin{table}[t]")
    tab_lines.append("  \\centering")
    tab_lines.append(f"  \\input{{{(TABLES/'table_quadratic_sweep_summary.tex').as_posix()}}}")
    tab_lines.append("  \\caption{Ill-conditioned quadratic diagnostic sweep summary.}")
    tab_lines.append("  \\label{tab:quadratic}")
    tab_lines.append("\\end{table}")

    (OUT / "INCLUDE_TABLES.tex").write_text("\n".join(tab_lines), encoding="utf-8")
    print("[latex_snippets] wrote INCLUDE_FIGURES.tex and INCLUDE_TABLES.tex")


if __name__ == "__main__":
    main()
