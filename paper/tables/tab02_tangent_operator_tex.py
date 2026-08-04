"""Table II renderer — reads ONLY tabdata/tab02_tangent_operator.json."""
import _tabdata as td

STEM = "tab02_tangent_operator"
AXES = ("b", "q", "chi_Ay", "chi_By")

if __name__ == "__main__":
    d = td.load(STEM)
    rows = [[td.axis(a), td.pow10(d["rows"][a]["h"]), td.bound(d["rows"][a]["rel_fd"])]
            for a in AXES]
    header = [r"Axis $\theta_k$", r"$h$",
              r"Eq.~\eqref{eq:tangent} vs.\ finite differences"]
    p = td.write_tex(STEM, header, rows)
    print(f"[{STEM}] wrote {p}")
