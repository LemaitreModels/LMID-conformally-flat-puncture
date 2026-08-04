"""Table I data — the production parameter box and the shipped build configuration.

Recomputes (instantly, no solver, no corpus) from the two canonical sources:
  * ``pipeline.production_box``  — every box edge, the Smolyak level, the spatial grid,
    the enhanced axis set (the module that exists precisely so these are declared once);
  * ``parametric.parametric_nd_2c.smolyak_points`` — the unique sparse-grid node count,
    which follows from the dimension and the level alone.

So the table cannot drift from the box the corpora were built on: retargeting an edge in
``production_box`` and re-running ``make tables`` moves the paper's number with it.
"""
from lm.initial_data.parametric.parametric_nd_2c import smolyak_points
from lm.initial_data.pipeline import production_box as pb

import _tabdata as td

STEM = "tab01_production_box"


def build():
    d4 = {a["name"]: a for a in pb.aligned_box()}
    d8 = {a["name"]: a for a in pb.spin8_box()}
    level = pb.SMOLYAK_LEVEL
    return {
        "b": [d4["b"]["min"], d4["b"]["max"]],
        "q": [d4["q"]["min"], d4["q"]["max"]],
        "chi": [-pb.CHI_MAX, pb.CHI_MAX],
        "aligned_axes": list(pb.ALIGNED_SPIN_AXES),
        "spin8_axes": list(pb.SPIN8_AXES),
        "inplane_axes": [a for a in pb.SPIN8_AXES if a not in pb.ALIGNED_SPIN_AXES],
        "level": level,
        "grid": list(pb.PROD_GRID),
        "models": [
            {"d": len(d4), "nodes": smolyak_points(len(d4), level)},
            {"d": len(d8), "nodes": smolyak_points(len(d8), level)},
        ],
    }


if __name__ == "__main__":
    out = build()
    p = td.dump(STEM, out)
    print(f"[{STEM}] wrote {p}  "
          f"(nodes {', '.join(str(m['nodes']) for m in out['models'])})")
