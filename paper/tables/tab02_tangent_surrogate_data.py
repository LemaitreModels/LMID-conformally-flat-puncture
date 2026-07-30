"""Table II data — the exposed surrogate gradient against both references.

Recomputes from the solver (~15 s: two aligned-spin interpolants plus the reference
certified solves; no model corpus, no cluster) via the canonical producer
``lm.initial_data.pipeline.run_tangent_verification.surrogate_tangents``.
"""
import _tabdata as td
from lm.initial_data.pipeline.run_tangent_verification import surrogate_tangents

STEM = "tab02_tangent_surrogate"

if __name__ == "__main__":
    print(f"[{STEM}] recomputing ...")
    out = surrogate_tangents()
    p = td.dump(STEM, out)
    print(f"[{STEM}] wrote {p}  ({out['wall_clock_s']:.1f}s)")
