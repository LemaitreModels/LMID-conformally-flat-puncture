"""Table I renderer — reads ONLY tabdata/tab01_production_box.json."""
import _tabdata as td

STEM = "tab01_production_box"


def rng(lo, hi, fmt="{:g}"):
    return rf"$[{fmt.format(lo)},\,{fmt.format(hi)}]$"


if __name__ == "__main__":
    d = td.load(STEM)
    b_lo, b_hi = d["b"]
    na, nb, nphi = d["grid"]

    # both quasi-circular models sample the same box on the same grid at the same level,
    # so the table is a single headerless quantity/value column of what they SHARE; what
    # distinguishes them (active spin components, dimension, node count) is in the text.
    rows = [
        [r"puncture separation $D=2b$ $[M]$", rng(2 * b_lo, 2 * b_hi)],
        [r"mass ratio $q=m_A/m_B$", rng(*d["q"])],
        [r"spin components $\chi^{A}_{i},\chi^{B}_{i}$", rng(*d["chi"])],
        [r"total bare mass $M=m_A+m_B$", "$1$"],
        [r"Smolyak level $\ell$", f"${d['level']}$"],
        [r"spatial grid $(N_A,N_B,N_\phi)$", rf"$({na},{nb},{nphi})$"],
        [r"gradient-enhanced axes $\mathcal{E}$", r"$\{\chi^{A}_{y},\chi^{B}_{y}\}$"],
    ]
    p = td.write_tex(STEM, None, rows)
    print(f"[{STEM}] wrote {p}")
