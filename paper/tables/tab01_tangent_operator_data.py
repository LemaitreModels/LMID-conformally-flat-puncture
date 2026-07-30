"""Table I data — Eq. (tangent) against finite differences of the certified 3-D solve.

Recomputes from the solver (~1 s, no model corpus, no cluster) via the canonical
producer ``lm.initial_data.pipeline.run_tangent_verification.operator_tangents``.
"""
import _tabdata as td
from lm.initial_data.pipeline.run_tangent_verification import operator_tangents

STEM = "tab01_tangent_operator"

if __name__ == "__main__":
    print(f"[{STEM}] recomputing ...")
    out = operator_tangents()
    p = td.dump(STEM, out)
    print(f"[{STEM}] wrote {p}  ({out['wall_clock_s']:.1f}s)")
