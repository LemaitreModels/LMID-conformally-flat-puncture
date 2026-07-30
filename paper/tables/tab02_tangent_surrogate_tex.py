"""Table II renderer — reads ONLY tabdata/tab02_tangent_surrogate.json."""
import _tabdata as td

STEM = "tab02_tangent_surrogate"
AXES = ("b", "q", "chi_A", "chi_B")

if __name__ == "__main__":
    d = td.load(STEM)
    rows = [[td.axis(a), td.pow10(d["rows"][a]["h"]),
             td.sci(d["rows"][a]["rel_fd"]), td.sci(d["rows"][a]["rel_ift"])]
            for a in AXES]
    header = [r"Axis $\theta_k$", r"$h$", r"vs.\ finite differences",
              r"vs.\ Eq.~\eqref{eq:tangent}"]
    p = td.write_tex(STEM, header, rows)
    print(f"[{STEM}] wrote {p}")
