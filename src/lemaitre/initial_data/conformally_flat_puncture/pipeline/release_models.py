"""Publish the shipped models on Zenodo, as a new version of one record.

Two steps, both run where the model files are (a cluster: the files are hundreds
of MB, so they go from there to Zenodo, not through a laptop)::

    # 1. stage: checksums, manifest, deposit metadata, README -- no network
    python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.release_models \\
        stage --version 1.0.0 --out ~/lmid-release-1.0.0

    # 2. upload a DRAFT (first time: --new; afterwards: a new version of the record)
    export ZENODO_TOKEN=...      # zenodo.org > Applications > Personal access token,
                                 # scopes deposit:write and deposit:actions
    python -m ...release_models upload --stage ~/lmid-release-1.0.0 --new
    python -m ...release_models upload --stage ~/lmid-release-1.1.0 \\
        --concept <concept record id>

``upload`` never publishes.  It prints the draft's URL: review the files and the
metadata there and press Publish.  ``--sandbox`` sends the draft to
sandbox.zenodo.org instead, to rehearse; a sandbox token is separate.  After the
first publication, set ``models.ZENODO_CONCEPT_RECORD`` to the concept record id
that ``upload --new`` prints: from then on :func:`..models.load` finds every later
version by itself.

What a version holds
--------------------
* the POD model files (``production_model.pod_stem(dim) + ".npz"``), unchanged;
* ``manifest.json`` -- for each model its file, SHA-256, size, dimension, axis
  names, box, POD rank and grid, plus the three keys a reader checks before it
  loads anything (``models.compatibility()``: model-file format, scheme, momentum
  prescription) and the package version and commit that built the files;
* ``README.md`` -- what the files are and how to load them.

The stage directory also holds ``sources.json`` (the local paths the files came
from) and ``zenodo.json`` (the deposit metadata).  Neither is uploaded as a file:
``sources.json`` carries cluster paths, which stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import production_model as pm
from .. import __version__
from .. import models as M
from ..paths import reports_root

REPO = "https://github.com/LemaitreModels/LMID-conformally-flat-puncture"
PAPER_ARXIV = "2609.10173"
DOCS = "https://lemaitre.readthedocs.io/projects/lmid-conformally-flat-puncture/"
DIMS = {"4d": 4, "8d": 8}


def default_path(name: str) -> Path:
    """Where the producers leave the shipped POD model of ``name``."""
    return Path(reports_root()) / "P2" / "models_chi" / (pm.pod_stem(DIMS[name]) + ".npz")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        raw = data["meta_json"]
        return json.loads(raw.item() if hasattr(raw, "item") else str(raw))


def _git_commit() -> str:
    """The commit of the package doing the release (not of the current directory)."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _model_entry(name: str, path: Path) -> dict:
    meta = _meta(path)
    need = M.compatibility()
    if meta.get("qc_prescription") != need["qc_prescription"]:
        raise SystemExit(f"{path}: built with momentum prescription "
                         f"{meta.get('qc_prescription')!r}, the code uses "
                         f"{need['qc_prescription']!r}; rebuild before releasing")
    from ..solver.operators_abt import scheme_of
    scheme = scheme_of(meta.get("bmap", "cos"), meta.get("radial", "lobatto"))
    if scheme != need["scheme"]:
        raise SystemExit(f"{path}: on grid {scheme!r}, the code ships {need['scheme']!r}")
    with np.load(path, allow_pickle=False) as data:
        rank = int(np.asarray(data["Phi"]).shape[1])
        box = np.asarray(data["axes"], dtype=float).tolist()
    return {"file": path.name, "sha256": _sha256(path), "size": path.stat().st_size,
            "dim": DIMS[name], "axis_names": list(meta["axis_names"]), "box": box,
            "fixed": meta.get("fixed") or {}, "rank": rank,
            "grid": [int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])],
            "kind": meta.get("kind"), "built_at_commit": meta.get("git_commit")}


def _readme(manifest: dict) -> str:
    rows = "\n".join(
        f"| `{n}` | `{e['file']}` | {e['size'] / 2**20:.0f} MiB | "
        f"{', '.join(e['axis_names'])} | {e['rank']} |"
        for n, e in manifest["models"].items())
    return f"""# Conformally flat puncture initial data: shipped models, version {manifest['version']}

Parametric models of binary-black-hole puncture initial data (quasi-circular,
conformally flat Bowen-York punctures): the conformal-factor correction u on a
single-patch spectral grid, as a differentiable function of the binary's
parameters. Each file is a gradient-enhanced sparse-grid interpolant of POD
coefficients. A few Newton steps of the package's solver turn its prediction into
certified initial data.

| name | file | size | parameters | POD rank |
|---|---|---|---|---|
{rows}

## Use

Install the package (see the documentation), then

```python
from lemaitre.initial_data.conformally_flat_puncture import models
model = models.load("8d")      # downloads this record once, then caches it
```

The package finds the newest version of this record that it can read. Pin this
one with `models.load("8d", version="{manifest['version']}")`.

`manifest.json` lists each file's SHA-256, its parameter box, grid and rank, and
the keys the package checks before it loads anything (`model_format`, `scheme`,
`qc_prescription`).

- Code: {REPO} (commit `{manifest['git_commit']}`, package version {manifest['package_version']})
- Documentation: {DOCS}
- Paper: https://arxiv.org/abs/{PAPER_ARXIV}
"""


def _zenodo_metadata(manifest: dict) -> dict:
    return {"metadata": {
        "title": "Conformally flat puncture initial data: parametric models "
                 "(LMID-conformally-flat-puncture)",
        "upload_type": "dataset",
        "version": manifest["version"],
        "description": (
            "<p>The shipped parametric models of the LMID-conformally-flat-puncture "
            "package: binary-black-hole puncture initial data (quasi-circular, "
            "conformally flat Bowen&ndash;York punctures) as a differentiable function "
            "of separation, mass ratio and spins. The package downloads these files "
            "itself; see README.md and the documentation.</p>"),
        "creators": [
            {"name": "De Ceuster, Frederik", "affiliation": "KU Leuven"},
            {"name": "Li, Tjonnie G.-F.", "affiliation": "KU Leuven"}],
        "license": "cc-by-4.0",
        "access_right": "open",
        "keywords": ["numerical relativity", "initial data", "binary black holes",
                     "reduced-order model", "spectral methods"],
        "related_identifiers": [
            {"relation": "isSupplementTo", "identifier": f"arXiv:{PAPER_ARXIV}",
             "scheme": "arxiv"},
            {"relation": "isSupplementTo", "identifier": REPO, "scheme": "url"},
            {"relation": "isDocumentedBy", "identifier": DOCS, "scheme": "url"}],
    }}


def stage(args) -> None:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths = {n: default_path(n).resolve() for n in DIMS}
    for spec in args.model or []:
        n, _, p = spec.partition("=")
        if n not in DIMS or not p:
            raise SystemExit(f"--model takes NAME=PATH with NAME in {list(DIMS)}")
        paths[n] = Path(p).expanduser().resolve()
    if args.only:
        paths = {n: paths[n] for n in args.only}
    for n, p in paths.items():
        if not p.exists():
            raise SystemExit(f"{n}: no model at {p} (pass --model {n}=PATH)")
    manifest = {"version": args.version, **M.compatibility(),
                "package": "LMID-conformally-flat-puncture", "package_version": __version__,
                "git_commit": _git_commit(), "models": {}}
    for n, p in paths.items():
        print(f"[stage] {n}: hashing {p} ...", flush=True)
        manifest["models"][n] = _model_entry(n, p)
    (out / M.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(_readme(manifest))
    (out / "zenodo.json").write_text(json.dumps(_zenodo_metadata(manifest), indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps({n: str(p) for n, p in paths.items()},
                                                 indent=2) + "\n")
    for n, e in manifest["models"].items():
        print(f"[stage] {n}: {e['file']}  {e['size'] / 2**20:.1f} MiB  sha256 {e['sha256'][:16]}...")
    print(f"[stage] wrote {out}/{{{M.MANIFEST}, README.md, zenodo.json, sources.json}}")


# --------------------------------------------------------------------------
# upload (Zenodo deposit API)
# --------------------------------------------------------------------------
class _Zenodo:
    def __init__(self, sandbox: bool):
        self.api = ("https://sandbox.zenodo.org/api" if sandbox else M.ZENODO_API)
        self.token = os.environ.get("ZENODO_TOKEN")
        if not self.token:
            raise SystemExit("set ZENODO_TOKEN (scopes deposit:write, deposit:actions)")

    def call(self, method, url, body=None, *, raw=None, size=None):
        if not url.startswith("http"):
            url = self.api + url
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            data = raw
            headers["Content-Type"] = "application/octet-stream"
            headers["Content-Length"] = str(size)
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                text = r.read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {url}: HTTP {e.code}: {e.read().decode()[:500]}")
        return json.loads(text) if text else {}


def upload(args) -> None:
    st = Path(args.stage).expanduser()
    manifest = json.loads((st / M.MANIFEST).read_text())
    sources = json.loads((st / "sources.json").read_text())
    meta = json.loads((st / "zenodo.json").read_text())
    if args.new == (args.concept is not None):
        raise SystemExit("pass exactly one of --new and --concept ID")
    # Preflight before touching Zenodo: every model file must still be the one the
    # manifest describes (a rebuild between stage and upload would publish files
    # whose downloads then fail their checksum).
    for n, e in manifest["models"].items():
        p = Path(sources[n])
        if not p.is_file() or p.stat().st_size != e["size"] or _sha256(p) != e["sha256"]:
            raise SystemExit(f"{n}: {p} no longer matches the staged manifest; stage again")
    z = _Zenodo(args.sandbox)
    if args.new:
        dep = z.call("POST", "/deposit/depositions", {})
    else:
        latest = z.call("GET", f"/records/{args.concept}")
        nv = z.call("POST", f"/deposit/depositions/{latest['id']}/actions/newversion")
        dep = z.call("GET", nv["links"]["latest_draft"])
        for f in dep.get("files", []):              # a new version inherits the old files
            z.call("DELETE", f"/deposit/depositions/{dep['id']}/files/{f['id']}")
    bucket = dep["links"]["bucket"]
    uploads = [(st / M.MANIFEST), (st / "README.md")]
    uploads += [Path(sources[n]) for n in manifest["models"]]
    for p in uploads:
        size = p.stat().st_size
        print(f"[upload] {p.name} ({size / 2**20:.1f} MiB) ...", flush=True)
        with open(p, "rb") as f:
            z.call("PUT", f"{bucket}/{p.name}", raw=f, size=size)
    z.call("PUT", f"/deposit/depositions/{dep['id']}", meta)
    html = dep["links"].get("html") or f"{z.api.replace('/api', '')}/deposit/{dep['id']}"
    print(f"[upload] draft {dep['id']} (concept record {dep.get('conceptrecid')}) is ready, "
          f"NOT published.\n         Review and publish it at {html}")
    if args.new:
        print(f"[upload] after publishing, set models.ZENODO_CONCEPT_RECORD = "
              f"{dep.get('conceptrecid')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="checksums, manifest and metadata (no network)")
    s.add_argument("--version", required=True, help="the release version, e.g. 1.0.0")
    s.add_argument("--out", required=True, help="the stage directory to write")
    s.add_argument("--model", action="append", metavar="NAME=PATH",
                   help="a model file other than the producers' default")
    s.add_argument("--only", nargs="+", choices=list(DIMS), help="release only these")
    u = sub.add_parser("upload", help="upload a staged release as an unpublished draft")
    u.add_argument("--stage", required=True)
    u.add_argument("--new", action="store_true", help="create the record (first release)")
    u.add_argument("--concept", type=int, help="add a new version to this concept record")
    u.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    args = ap.parse_args(argv)
    {"stage": stage, "upload": upload}[args.cmd](args)


if __name__ == "__main__":                                       # pragma: no cover
    main(sys.argv[1:])
