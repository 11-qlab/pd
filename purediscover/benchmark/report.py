"""Report renderers for benchmark results (text / markdown / latex)."""
from __future__ import annotations

from purediscover.benchmark.results import BenchmarkResult, LawResult


class ReportRenderer:
    def __init__(self, result: BenchmarkResult):
        self.result = result

    # --- TEXT -------------------------------------------------------------
    def render_table(self) -> str:
        lines = ["=" * 100]
        lines.append(f"  {'law':25s}  {'method':20s}  {'conv':6s}  "
                     f"{'iter':5s}  {'pts':5s}  {'time':8s}")
        lines.append("=" * 100)
        for r in self.result.results:
            lines.append(self._row(r))
        return "\n".join(lines)

    def render_summary(self) -> str:
        lines = ["", "=" * 100, "  AGGREGATE SUMMARY", "=" * 100]
        lines.append(
            f"  {'method':25s}  {'conv':16s}  {'skipped':8s}  "
            f"{'median_iter':11s}  {'median_pts':11s}  {'median_t':9s}"
        )
        lines.append("  " + "-" * 94)
        for s in self.result.summaries:
            rate = f"{s.n_converged}/{s.n_applicable} " \
                   f"({s.convergence_rate*100:.0f}%)"
            mi = f"{s.median_iterations:.1f}" if s.median_iterations is not None else "—"
            mp = f"{s.median_points:.1f}" if s.median_points is not None else "—"
            mt = f"{s.median_time_s:.1f}s" if s.median_time_s is not None else "—"
            lines.append(
                f"  {s.method:25s}  {rate:16s}  {s.n_skipped:<8d}  "
                f"{mi:11s}  {mp:11s}  {mt:9s}"
            )
        lines.append("")
        lines.append(f"  total wall time: {self.result.wall_time_s:.1f}s")
        return "\n".join(lines)

    def render_full(self) -> str:
        return "\n".join([
            "BENCHMARK RESULT",
            f"  timestamp : {self.result.timestamp}",
            f"  laws      : {self.result.n_laws}",
            f"  noise     : {self.result.noise_frac * 100:.0f}%",
            f"  methods   : {self.result.method_names}",
            "",
            self.render_table(),
            self.render_summary(),
        ])

    # --- MARKDOWN ---------------------------------------------------------
    def render_markdown(self) -> str:
        lines = ["# Benchmark Result", ""]
        lines.append(f"**Timestamp:** {self.result.timestamp}  ")
        lines.append(f"**Laws:** {self.result.n_laws}  ")
        lines.append(f"**Noise:** {self.result.noise_frac * 100:.0f}%  ")
        lines.append(f"**Wall time:** {self.result.wall_time_s:.1f}s")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append("| Method | Conv. | Skipped | Median iter | Median pts | Median time |")
        lines.append("|---|---|---|---|---|---|")
        for s in self.result.summaries:
            rate = f"{s.n_converged}/{s.n_applicable} ({s.convergence_rate*100:.0f}%)"
            mi = f"{s.median_iterations:.1f}" if s.median_iterations is not None else "—"
            mp = f"{s.median_points:.1f}" if s.median_points is not None else "—"
            mt = f"{s.median_time_s:.1f}s" if s.median_time_s is not None else "—"
            lines.append(
                f"| {s.method} | {rate} | {s.n_skipped} | {mi} | {mp} | {mt} |"
            )

        lines.append("")
        lines.append("## Per-law")
        lines.append("")
        lines.append("| Law | Method | Conv. | Iter | Pts | Time |")
        lines.append("|---|---|---|---|---|---|")
        for r in self.result.results:
            mark = "✓" if r.converged else ("skip" if r.skipped else "✗")
            lines.append(
                f"| {r.law_name} | {r.method} | {mark} | "
                f"{r.iterations} | {r.points_used} | {r.time_s:.1f}s |"
            )
        return "\n".join(lines)

    # --- LATEX ------------------------------------------------------------
    def render_latex(self) -> str:
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Benchmark summary on "
                     f"{self.result.n_laws} physical laws at "
                     f"{self.result.noise_frac*100:.0f}\\% noise.}}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\hline")
        lines.append("Method & Conv. & Skipped & Median iter & "
                     "Median pts & Median time \\\\")
        lines.append("\\hline")
        for s in self.result.summaries:
            rate = f"{s.n_converged}/{s.n_applicable} ({s.convergence_rate*100:.0f}\\%)"
            mi = f"{s.median_iterations:.1f}" if s.median_iterations is not None else "---"
            mp = f"{s.median_points:.1f}" if s.median_points is not None else "---"
            mt = f"{s.median_time_s:.1f}\\,s" if s.median_time_s is not None else "---"
            lines.append(
                f"{s.method.replace('_', chr(92)+'_')} & {rate} & "
                f"{s.n_skipped} & {mi} & {mp} & {mt} \\\\"
            )
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _row(r: LawResult) -> str:
        if r.skipped:
            mark, detail = "skip", f"[{r.error}]"
        elif r.converged:
            mark, detail = "✓ ", ""
        else:
            mark, detail = "✗ ", f"[{r.error}]" if r.error else ""
        return (
            f"  {r.law_name:25s}  {r.method:20s}  {mark:6s}  "
            f"{r.iterations:>4d}  {r.points_used:>4d}  "
            f"{r.time_s:>7.2f}s  {detail}"
        )


__all__ = ["ReportRenderer"]
