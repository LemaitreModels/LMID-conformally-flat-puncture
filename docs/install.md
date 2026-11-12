# Installation

The package needs Python 3.10 or later and four libraries: `numpy`, `scipy`, `jax`
and `matplotlib`. Everything runs in double precision on a CPU; no GPU is needed.

It lives in the `lemaitre` namespace, `lemaitre.initial_data.conformally_flat_puncture`,
which spans three distributions: the namespace root `lemaitre`, the `initial_data`
level `LM-initial-data`, and this package. Install all three from one checkout of
the family repository:

```bash
git clone https://github.com/LemaitreModels/Lemaitre.git
cd Lemaitre
git submodule update --init LM-initial-data
git -C LM-initial-data submodule update --init LMID-conformally-flat-puncture

python -m pip install numpy scipy jax matplotlib
for d in . LM-initial-data LM-initial-data/LMID-conformally-flat-puncture; do
  python -m pip install -e "$d" --config-settings editable_mode=compat --no-deps
done
```

Two details of the install loop matter, and both fail silently when left out:

- **`--no-deps`, in this order.** The two namespace parents are also on PyPI, so
  without it pip installs them from there rather than from your checkout.
- **`editable_mode=compat`.** The default editable mode turns the namespace levels
  into bare directories, and then `lemaitre.initial_data.conformally_flat_puncture`
  resolves only through a direct import.

Check the install:

```bash
python -c "import lemaitre as lm; print(lm.initial_data.conformally_flat_puncture.__file__)"
```

The path it prints should be inside your checkout.

**The models download themselves.** [Evaluate a model](evaluate.md) and [Refine a
model prediction](refine.md) need a model file; the first `models.load(...)` fetches
it from Zenodo into `~/.cache/lemaitre/conformally_flat_puncture` and checks its
checksum. [Available models](MODELS.md) lists them and the settings for the cache and
for offline use. [Solve one binary](solve.md) needs no download.

**To run the tests or rebuild the paper**, also install `pytest` and `sympy` (the
`[dev]` extra) and see [Reproduce the paper](DATA.md).
