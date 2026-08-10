"""LM-initial-data — WHICH model is shipped, WHAT it stores, and what that costs.

Single source of truth for the shipped surrogate, the sibling of
:mod:`production_box` (which fixes the parameter *box*; this fixes the *model* on
it).  Producers, figure-data scripts, tests and the paper all read the model
identity, the shipped POD ranks and the stored-memory accounting from here.  If a
number about "the model" appears anywhere, it must come from this module.

Why this module exists
----------------------
Five independent things used to answer "which model?" differently, and the answers
disagreed in ways that reached the paper:

1.  Seven producer modules each carried their own copy of the POD memory formula,
    every one of them keyed on the FULL dimension ``d`` rather than on the number
    of ENHANCED axes the shipped model actually uses.
2.  Nothing named the shipped model.  Its identity lived in ``.npz`` filenames, a
    ``BR_8D_ENHANCED`` constant inside one figure script, and prose comments.
3.  The shipped POD rank was a CLI default (``--rank 75``) that drifted to 250/500
    without anything recording the change.
4.  The stored artifact carried ``d`` tangent blocks per node although the shipped
    interpolant reads only the ``n_enh`` enhanced ones (see :func:`stored_blocks`),
    so every memory number was inflated by ``(1+d+npair)/(1+n_enh+npair)``.
5.  Figure data recorded no provenance, so a panel could be measured on one model
    while its caption described another -- which happened twice (the 8-D field
    panel until 2026-08-02, and fig06 for a whole revision).

The named families in :data:`FAMILIES` exist so that each historical number stays
reproducible and *labelled*: ``gradient_all`` reproduces the pre-2026-08 paper
text, ``cross_all`` reproduces the pre-2026-08 Fig. 5 stars, and ``shipped`` is the
model the paper now describes.  Never add an unnamed variant.

Pure stdlib on purpose: the figure-data scripts and the paper tests import this
without pulling in jax.
"""
from __future__ import annotations

from itertools import combinations

from . import production_box as pb

# ---------------------------------------------------------------------------
# 1.  Identity of the shipped model
# ---------------------------------------------------------------------------
ENHANCED_AXES = pb.ALIGNED_SPIN_AXES
"""The enhanced axis set E: the two spin components carrying certified tangents.

Both shipped models enhance exactly these two axes -- the aligned (orbital
angular momentum) spin components -- and no others.  ``sec:model:enhanced`` of the
paper is the measurement behind that choice: enhancing a single axis always beats
the value interpolant, enhancing several axes with tangents alone *degrades* it
whichever axes are chosen, and the two-axis case is rescued only by adding the
bilinear cross term below.  The names are the same in both boxes; the *indices*
are not (see :func:`enhanced_indices`).
"""

CROSS = True
"""The shipped model carries the one bilinear cross term of the enhanced pair.

``C(n_enh, 2) = C(2, 2) = 1`` mixed second partial, so the full bilinear (= full
tensor product, hence exact) treatment of the enhanced subspace.
"""

SHIPPED_RANK = {4: 250, 8: 500}
"""POD truncation rank of the shipped compressed model, per dimension.

Set to the rank at which the Fig. 5 ladder shows the compression is no longer the
limiting error -- i.e. the bare-guess residual has saturated at its full-rank
value, so the truncation costs neither accuracy nor a Newton step.

History (do not resurrect): the first shipped bases held 76 (4-D) and 359 (8-D)
modes, well below that knee, and the loaders truncate *downward only*, so asking
for a higher rank silently clamped.  Raising the ranks required rebuilding the
cross POD bases from the full Hermite corpora.
"""

N_NODES = {4: 1105, 8: 15713}
"""Solver nodes in the isotropic Smolyak level-``pb.SMOLYAK_LEVEL`` corpus."""

N_FEAT = 11520
"""Spatial degrees of freedom per stored field (``Na * Nb * Nphi`` of the
production grid), measured from the corpora."""

BYTES_PER_FLOAT = 8
"""All stored arrays are float64."""


def enhanced_indices(dim: int) -> tuple[int, ...]:
    """Global axis indices of :data:`ENHANCED_AXES` in the ``dim``-D production box.

    The enhanced axes have the same *names* in both boxes but different indices --
    ``(2, 3)`` in the 4-D aligned box and ``(3, 6)`` in the 8-D general-spin box --
    because the six spin components sit between ``q`` and the aligned pair.  Index
    an enhanced-axis array with these, never with a literal.
    """
    box = pb.aligned_box() if dim == 4 else pb.spin8_box()
    names = [a["name"] for a in box]
    return tuple(names.index(n) for n in ENHANCED_AXES)


def n_enhanced() -> int:
    """Number of enhanced axes (2)."""
    return len(ENHANCED_AXES)


def n_pairs() -> int:
    """Number of stored bilinear cross terms, ``C(n_enh, 2)`` (1), or 0 without cross."""
    return len(list(combinations(range(n_enhanced()), 2))) if CROSS else 0


# ---------------------------------------------------------------------------
# 2.  Stored-memory accounting
# ---------------------------------------------------------------------------
FAMILIES = ("value", "gradient_all", "cross_all", "shipped")
"""Every model family whose memory has ever been quoted, so each stays labelled.

``value``         value only, one field per node -- the value-only interpolant.
``gradient_all``  value + all ``d`` tangents, no cross.  Reproduces the paper's
                  pre-2026-08 corpus numbers (486 MiB / 12.1 GiB).
``cross_all``     value + all ``d`` tangents + the cross.  Reproduces the
                  pre-2026-08 Fig. 5 bare stars (583 MiB / 13.5 GiB).  This is
                  what the artifact physically held, dead blocks included.
``shipped``       value + the ``n_enh`` ENHANCED tangents + the cross.  The model
                  the paper describes and the figures plot.
"""


def stored_blocks(family: str = "shipped", dim: int | None = None) -> int:
    """Fields stored per node by ``family`` (the multiplier on ``N * nfeat``).

    ``shipped`` is ``1 + n_enh + npair = 4`` and is INDEPENDENT of ``d``: the
    cross-enhanced interpolant reads ``dU_nodes`` only at the enhanced axes
    (``hermite_smolyak_cross.HermiteCrossSolutionND.evaluate``, and its jax twin,
    both do ``{e: take(dU_nodes, e) for e in enhanced}``), so the other ``d -
    n_enh`` tangent blocks are provably never read.  Storing them was the
    inflation of confusion 4 above; the ``slim=True`` layout of
    ``hermite_smolyak_pod_cross.PODHermiteSmolyakCross.save`` removes them losslessly.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if family == "value":
        return 1
    if family == "shipped":
        return 1 + n_enhanced() + n_pairs()
    if dim is None:
        raise ValueError(f"family {family!r} is d-dependent; pass dim=")
    return 1 + dim + (n_pairs() if family == "cross_all" else 0)


def blocks_of_model(model) -> int:
    """Fields stored per node by an ACTUAL model object — the general form.

    Producers must use this rather than ``1 + d + npair``: memory is set by what is
    stored, which is the value, one tangent per ENHANCED axis, and one field per
    cross pair.  Keying on ``d`` was the single arithmetic error behind every
    inflated memory number in the pipeline, and it is silent — it gives the right
    answer only when every axis happens to be enhanced.

    Accepts anything exposing ``enhanced`` (and optionally ``cross_pairs_global``):
    the Hermite-Smolyak models, their POD re-encodings, and the cross siblings.
    """
    n_enh = len(tuple(model.enhanced))
    npair = len(tuple(getattr(model, "cross_pairs_global", ()) or ()))
    return 1 + n_enh + npair


def bare_bytes_of(N: int, nfeat: int, blocks: int) -> float:
    """Uncompressed corpus bytes for an explicit block count (see :func:`blocks_of_model`)."""
    return float(BYTES_PER_FLOAT * N * blocks * nfeat)


def pod_bytes_of(r: int, N: int, nfeat: int, blocks: int) -> float:
    """Rank-``r`` POD bytes for an explicit block count (see :func:`blocks_of_model`)."""
    return float(BYTES_PER_FLOAT * (nfeat * r + N * blocks * r + nfeat))


def bare_bytes(dim: int, family: str = "shipped", *, N: int | None = None,
               nfeat: int = N_FEAT) -> float:
    """Stored bytes of the UNCOMPRESSED corpus: ``8 * N * blocks * nfeat``."""
    N = N_NODES[dim] if N is None else N
    return float(BYTES_PER_FLOAT * N * stored_blocks(family, dim) * nfeat)


def pod_bytes(r: int, dim: int, family: str = "shipped", *, N: int | None = None,
              nfeat: int = N_FEAT) -> float:
    """Stored bytes of the rank-``r`` POD model.

    ``Phi (nfeat*r) + one coefficient block per stored field (N*blocks*r) + mean
    (nfeat)``, float64.  Matches the on-disk ``.npz`` to <0.5%.
    """
    N = N_NODES[dim] if N is None else N
    return float(BYTES_PER_FLOAT * (nfeat * r + N * stored_blocks(family, dim) * r + nfeat))


def compression_factor(dim: int, r: int | None = None, family: str = "shipped") -> float:
    """Bare corpus / rank-``r`` POD, both in ``family``'s accounting."""
    r = SHIPPED_RANK[dim] if r is None else r
    return bare_bytes(dim, family) / pod_bytes(r, dim, family)


# ---------------------------------------------------------------------------
# 3.  Artifact naming
# ---------------------------------------------------------------------------
_BOX_STEM = {4: "d4qc", 8: "spin8qc"}


def model_stem(dim: int) -> str:
    """Filename stem of the uncompressed shipped Hermite-Smolyak cross model."""
    enh = "-".join(ENHANCED_AXES)
    stem = f"hermite_smolyak_{_BOX_STEM[dim]}_L{pb.SMOLYAK_LEVEL}_enh-{enh}"
    return stem + ("_cross" if CROSS else "")


def pod_stem(dim: int, rank: int | None = None) -> str:
    """Filename stem of the shipped rank-``rank`` POD artifact.

    The rank is in the name so two ranks never clobber each other.  The storage
    LAYOUT is deliberately not in the name -- it is recorded inside the file as
    ``meta['dU_layout']`` and both layouts load, so a name cannot go stale against
    its contents.
    """
    rank = SHIPPED_RANK[dim] if rank is None else rank
    return f"pod_{model_stem(dim)}_r{int(rank)}"


def describe(dim: int) -> str:
    """One-line human summary, for driver banners and figure provenance."""
    r = SHIPPED_RANK[dim]
    return (f"{dim}-D shipped: cross-enhanced on {'+'.join(ENHANCED_AXES)} "
            f"(n_enh={n_enhanced()}, npair={n_pairs()}), Smolyak L={pb.SMOLYAK_LEVEL}, "
            f"N={N_NODES[dim]}, POD r={r}, {stored_blocks('shipped')} blocks/node, "
            f"{bare_bytes(dim)/2**20:.1f} MiB -> {pod_bytes(r, dim)/2**20:.1f} MiB "
            f"({compression_factor(dim):.1f}x)")


# ---------------------------------------------------------------------------
# 4.  Self-report (keeps docs/MODELS.md honest: regenerate, never hand-edit)
# ---------------------------------------------------------------------------
def table() -> str:
    """Markdown table of every family's stored memory, both dimensions."""
    hdr = ("| family | blocks/node | 4-D bare | 4-D POD | 8-D bare | 8-D POD |\n"
           "|---|---|---|---|---|---|\n")
    rows = []
    for fam in FAMILIES:
        cells = []
        for dim in (4, 8):
            r = SHIPPED_RANK[dim]
            cells += [f"{bare_bytes(dim, fam)/2**20:,.0f} MiB",
                      f"{pod_bytes(r, dim, fam)/2**20:,.1f} MiB"]
        blocks = (stored_blocks(fam, 4), stored_blocks(fam, 8))
        b = str(blocks[0]) if blocks[0] == blocks[1] else f"{blocks[0]} / {blocks[1]}"
        mark = " **(shipped)**" if fam == "shipped" else ""
        rows.append(f"| `{fam}`{mark} | {b} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows)


if __name__ == "__main__":                                    # pragma: no cover
    for _dim in (4, 8):
        print(describe(_dim))
    print(f"\nPOD ranks: {SHIPPED_RANK}   (4-D and 8-D at their Fig. 5 saturation knee)\n")
    print(table())
