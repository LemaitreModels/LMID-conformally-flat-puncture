"""Shared test setup: float64 before collection, and one recorded negative result.

This file is deliberately small. It exists for two reasons, and the second is the
useful one: **the persistent JAX compilation cache that makes the curved sibling's
suite 2.2x faster does nothing here, and that was measured rather than assumed.**
Without this note the next person to compare the two leaves re-derives it.

Measured 2026-08-16 on leaf ``1a3d90d``, 8-core M-series laptop, peers on the box
(``uptime`` load 2.4-8.3 across the runs; the arms below were interleaved and the
warm arm ran at the *higher* load, so it is if anything flattered).

Selection ``pytest -m "not slow"`` -- 631 of 661 tests, the compile-diverse part of
the suite:

=========================================  ============  ==============
cache state                                wall          **user CPU**
=========================================  ============  ==============
disabled                                   ``329.21 s``  ``627.57 s``
cold, populating (1868 entries, 7.4 MB)    ``330.60 s``  ``640.14 s``
warm (1868 entries)                        ``329.36 s``  ``648.52 s``
=========================================  ============  ==============

Warm against uncached is ``+0.05 %`` wall and ``+3.3 %`` CPU: no win, a slight net
loss. The mechanism is not misconfigured -- 1868 entries is comparable to the
sibling's 1504 for a single file, so the cache filled and was read. There simply
is no compile time here to recover, and the disk deserialization costs about what
the sub-millisecond recompiles did.

The same holds on the heavy path. ``test_certified_polish_4d`` alone (140 s, the
second-costliest test in the suite): ``131.9 s`` uncached against ``139.5 s`` and
``137.9 s`` warm, producing all of **18 cache entries / 72 kB** -- that test runs a
handful of kernels very many times, so there was nothing to cache in the first
place.

**The sibling's other lever does not transfer either.** Its conftest also sets
``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false``, worth ``1.44x`` there. On the same
selection here: ``334.93 s`` wall / ``631.82 s`` CPU against ``329.21``/``627.57``
uncached -- ``+1.7 %`` wall, i.e. a small loss. It is therefore not set. (It is not
a *correctness* problem: the three bit-for-bit gates --
``test_m0_block_is_the_frozen_axisymmetric_operator`` and both axisymmetric
reductions -- pass under the flag. It simply buys nothing, so the guardrail-2
question of whether a thread-count change may reorder the dense mode-apply never
has to be asked.)

**Why the sibling differs.** The curved suite is compile-bound: it re-traces large
expressions (the boosted-Kerr node fields, the vector Laplacian) in gates that do
few solves. This suite is solve-bound -- 75 % of its wall clock is five tests, and
``test_joint_held_out_below_1e8`` alone is 44 % of it. Caching compilations cannot
touch a bill that is spent inside GMRES, and neither can a thread-pool flag whose
gain is in dispatch overhead on small ops.

So the cache is **off by default and available in one environment variable**, for a
machine where the trade may differ -- a cluster node with slower compiles is the
plausible case. Set ``LM_JAX_CACHE=1`` to enable it. Two settings inside it are
measured, not chosen, and both come from the sibling's conftest, which is the
place to read the full argument:
``../../LMID-curved-puncture/tests/conftest.py``.

**And note the trap even the disabled default leaves.** If you ever turn it on,
every timing taken under pytest from then on is counting cache hits. Performance
work on this package is measured through the drivers or a standalone script, never
through the suite.

**Not done here, deliberately: session-scoped solved fixtures.** The review
proposed them as the lever on the wall clock; the durations say otherwise. The 30
module-scoped fixtures are already module-scoped, ``solver_3d.make_problem`` is a
grid build rather than a solve, and the expensive solves in different files run at
different slices and tolerances, so there is no shared object to hoist. Sharing
them would buy coupling, not minutes. The measured lever is the ``slow`` marker --
see ``make fast`` and the marker note in ``pyproject.toml``.
"""
from __future__ import annotations

import os
import pathlib


def _enable_float64() -> None:
    """``jax_enable_x64`` once, before collection imports the first test module.

    Not a fix for a missing call -- 64 modules under ``src/`` set it, and 15 test
    files repeat it. It is a guarantee about *order*: a conftest runs before any
    test module is imported, so the flag is on before anything can be traced,
    rather than on from whenever the first import that happens to set it lands.

    The per-file calls stay. They are idempotent, and each keeps its own module
    honest when it is imported outside pytest.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


def _enable_jax_compilation_cache() -> None:
    """Opt-in persistent compilation cache. Measured a no-op here -- see the module docstring.

    ``LM_JAX_CACHE=1`` turns it on; ``LM_JAX_CACHE_DIR`` relocates it; an explicit
    ``JAX_COMPILATION_CACHE_DIR`` always wins, which is how a cluster job points it
    at node-local storage. The directory is shared with the curved sibling, which
    is deliberate: that package calls this leaf's solver, so the two genuinely
    compile some of the same kernels.
    """
    if not os.environ.get("LM_JAX_CACHE"):
        return
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return

    cache_dir = pathlib.Path(
        os.environ.get("LM_JAX_CACHE_DIR")
        or pathlib.Path.home() / ".cache" / "lemaitre" / "jax")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return                      # read-only home: run uncached rather than fail

    import jax

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # Measured, not chosen. The 1.0 s default caches NOTHING here and on the
    # sibling: the cost is thousands of sub-second compiles, so at the default the
    # directory stays empty while looking configured.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    # `jax_compilation_cache_max_size` is deliberately NOT set. Without `filelock`
    # every cache access raises and XLA downgrades that to a warning, so the cache
    # silently stores nothing -- found on fys-kuleuven, 0 entries there against
    # 1504 locally from the same code. With `filelock` it doubles the files and
    # slows the populating run for no warm-run gain. There is nothing to protect:
    # this suite's whole cache is 7.4 MB, and deleting the directory is a complete
    # and safe recovery.


_enable_float64()
_enable_jax_compilation_cache()
