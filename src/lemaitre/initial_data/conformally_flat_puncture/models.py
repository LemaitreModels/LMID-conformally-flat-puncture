"""The shipped parametric models: find, download, check and load them.

The models are too large for the repository (the 8-D one is about 300 MB), so they
are published on Zenodo, one record with a new version for every rebuild.  Each
version holds the model files and a ``manifest.json`` that says which code can read
them.  :func:`load` picks the newest version this code can read, downloads the file
once into a local cache, checks its checksum, and returns the model::

    from lemaitre.initial_data.conformally_flat_puncture import models

    model = models.load("8d")                # evaluate / evaluate_jax work at once
    model = models.load("8d", solver=True)   # evaluate_polished works too
    prob = models.problem(model)             # the grid, for solver_3d.evaluate_field

A version is **compatible** when its manifest names the model-file format, the
discretisation (``production_model.SCHEME``) and the quasi-circular momentum
prescription (``quasicircular.QC_PRESCRIPTION``) this code uses.  A newer version
that needs newer code is skipped with a warning, so an old install keeps working.
Pin a version (``version="1.0.0"``) to reproduce a result exactly; the paper does.

Configuration, all optional:

* ``LMID_MODEL_CACHE`` — the cache directory (default
  ``~/.cache/lemaitre/conformally_flat_puncture``);
* ``LMID_OFFLINE=1`` — never touch the network; use the newest cached version.
  Without it the model file is cached but finding the newest version still asks
  Zenodo on every call;
* ``LMID_ZENODO_RECORD`` and ``LMID_ZENODO_API`` — another record or server
  (``https://sandbox.zenodo.org/api`` to test a release before publishing it).

Standard library only for the download (``urllib``), so this module adds no
dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

#: Zenodo's REST API.
ZENODO_API = "https://zenodo.org/api"

#: The Zenodo *concept* record of the shipped models: the id that always resolves to
#: the newest version (``pipeline/release_models.py upload`` prints it when it creates
#: the first deposit).  Version 1.0.0, the models of the paper, is
#: doi:10.5281/zenodo.24219765.
ZENODO_CONCEPT_RECORD: Optional[int] = 24219764

#: The model-file format this code reads.  Bump it only when a change to the
#: ``.npz`` layout makes old code unable to read new files.
MODEL_FORMAT = 1

#: The name of the per-version manifest file.
MANIFEST = "manifest.json"

#: The shipped models, by the name :func:`load` takes.
NAMES = ("4d", "8d")

_TIMEOUT = 60
_CHUNK = 1 << 20


class ModelUnavailable(RuntimeError):
    """No compatible model release could be found or downloaded."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def cache_dir() -> Path:
    """The local cache: ``$LMID_MODEL_CACHE``, else ``~/.cache/lemaitre/conformally_flat_puncture``."""
    env = os.environ.get("LMID_MODEL_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "lemaitre" / "conformally_flat_puncture"


def _api() -> str:
    return os.environ.get("LMID_ZENODO_API", ZENODO_API).rstrip("/")


def _concept() -> Optional[int]:
    env = os.environ.get("LMID_ZENODO_RECORD")
    return int(env) if env else ZENODO_CONCEPT_RECORD


def _offline() -> bool:
    return os.environ.get("LMID_OFFLINE", "").strip() not in ("", "0", "false", "False")


def compatibility() -> dict:
    """What a manifest must say for this code to read the models it lists."""
    from .pipeline import production_model as pm
    from .parametric.quasicircular import QC_PRESCRIPTION
    return {"model_format": MODEL_FORMAT, "scheme": pm.SCHEME,
            "qc_prescription": QC_PRESCRIPTION}


def is_compatible(manifest: dict) -> bool:
    """Whether this code can read the models a manifest lists."""
    need = compatibility()
    return all(manifest.get(k) == v for k, v in need.items())


# --------------------------------------------------------------------------
# Zenodo
# --------------------------------------------------------------------------
def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "lemaitre-lmid-models"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:     # follows redirects
        return json.load(r)


def _files_of(record: dict) -> dict:
    """``{key: {"size", "checksum", "url"}}`` of a Zenodo record, in either of the
    two shapes the API serves (a list, or ``{"entries": {...}}``)."""
    files = record.get("files") or []
    if isinstance(files, dict):
        files = list((files.get("entries") or {}).values())
    out = {}
    for f in files:
        key = f["key"]
        url = (f.get("links") or {}).get("self") or (
            f"{_api()}/records/{record['id']}/files/{key}/content")
        if not url.endswith("/content") and "/files/" in url and "/api/" in url:
            url = url + "/content"
        out[key] = {"size": int(f.get("size", -1)), "checksum": f.get("checksum", ""),
                    "url": url}
    return out


def _release_of(record: dict) -> dict:
    meta = record.get("metadata") or {}
    return {"record": int(record["id"]), "version": meta.get("version"),
            "published": meta.get("publication_date"), "doi": record.get("doi"),
            "files": _files_of(record)}


def releases() -> list:
    """Every published version of the model record, newest first.

    Each entry: ``record`` (Zenodo id), ``version``, ``published``, ``doi`` and
    ``files`` (``{name: {size, checksum, url}}``).
    """
    concept = _concept()
    if concept is None:
        raise ModelUnavailable(
            "the models have not been published on Zenodo yet "
            "(models.ZENODO_CONCEPT_RECORD is None); load a local file with "
            "models.load(path=...) or set LMID_ZENODO_RECORD")
    latest = _get_json(f"{_api()}/records/{concept}")       # redirects to the newest
    out, url = [], f"{_api()}/records/{latest['id']}/versions?size=100&sort=version"
    while url:
        page = _get_json(url)
        out += [_release_of(h) for h in page["hits"]["hits"]]
        url = (page.get("links") or {}).get("next")
    return out


def _manifest(release: dict) -> dict:
    """The manifest of a release, cached beside its files (published records are
    immutable, so a cached manifest never goes stale)."""
    path = cache_dir() / str(release["record"]) / MANIFEST
    if not path.exists():
        entry = release["files"].get(MANIFEST)
        if entry is None:
            raise ModelUnavailable(f"Zenodo record {release['record']} has no {MANIFEST}")
        _download(entry, path)
    return json.loads(path.read_text())


def _cached_releases() -> list:
    """Releases already in the cache, newest record first (for offline use)."""
    root = cache_dir()
    out = []
    if root.is_dir():
        for d in root.iterdir():
            m = d / MANIFEST
            if d.name.isdigit() and m.exists():
                out.append({"record": int(d.name), "manifest": json.loads(m.read_text())})
    return sorted(out, key=lambda r: r["record"], reverse=True)


# --------------------------------------------------------------------------
# download and check
# --------------------------------------------------------------------------
def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_ok(path: Path, info: dict) -> bool:
    """Whether a cached model file exists and matches its manifest entry (size and
    SHA-256).  Hashing the 8-D file takes about a second."""
    if not path.exists():
        return False
    if info.get("size", -1) >= 0 and path.stat().st_size != info["size"]:
        return False
    return not info.get("sha256") or _hash(path, "sha256") == info["sha256"]


def _download(entry: dict, dest: Path, sha256: Optional[str] = None) -> Path:
    """Download ``entry`` to ``dest`` through a temporary file, then check the size,
    Zenodo's own checksum and, if given, the manifest's SHA-256.  A file that fails
    a check is deleted, never left in the cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    try:
        req = urllib.request.Request(entry["url"], headers={"User-Agent": "lemaitre-lmid-models"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            shutil.copyfileobj(r, out, _CHUNK)
        tmp = Path(tmp)
        if entry.get("size", -1) >= 0 and tmp.stat().st_size != entry["size"]:
            raise ModelUnavailable(f"{dest.name}: downloaded {tmp.stat().st_size} bytes, "
                                   f"expected {entry['size']}")
        algo, _, want = (entry.get("checksum") or "").partition(":")
        if want and _hash(tmp, algo) != want:
            raise ModelUnavailable(f"{dest.name}: {algo} checksum mismatch")
        if sha256 and _hash(tmp, "sha256") != sha256:
            raise ModelUnavailable(f"{dest.name}: SHA-256 differs from the manifest")
        os.replace(tmp, dest)
        return dest
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def resolve(name: str = "8d", *, version: Optional[str] = None) -> dict:
    """The release that :func:`load` would use for ``name``, without downloading
    the model: ``{record, version, doi, manifest, entry}``.

    ``version=None`` takes the newest compatible version; a named version must be
    compatible or this raises.
    """
    if name not in NAMES:
        raise ValueError(f"unknown model {name!r}; the shipped models are {NAMES}")
    if _offline():
        for r in _cached_releases():
            m = r["manifest"]
            if (version is None or m.get("version") == version) and is_compatible(m) \
                    and name in m.get("models", {}) and _cached_ok(
                        cache_dir() / str(r["record"]) / m["models"][name]["file"],
                        m["models"][name]):
                return {"record": r["record"], "version": m.get("version"),
                        "doi": m.get("doi"), "manifest": m, "entry": None}
        raise ModelUnavailable(f"LMID_OFFLINE is set and no compatible {name!r} model "
                               f"is cached in {cache_dir()}")
    skipped = []
    for rel in releases():
        if version is not None and rel["version"] != version:
            continue
        m = _manifest(rel)
        if not is_compatible(m) or name not in m.get("models", {}):
            skipped.append(rel["version"])
            if version is not None:
                raise ModelUnavailable(
                    f"model release {version} needs {{model_format, scheme, "
                    f"qc_prescription}} = {[m.get(k) for k in compatibility()]}, "
                    f"this code reads {list(compatibility().values())}")
            continue
        if skipped:
            warnings.warn(f"newer model release(s) {skipped} need a newer version of "
                          f"this package; using {rel['version']}. Upgrade to use them.")
        return {"record": rel["record"], "version": rel["version"], "doi": rel["doi"],
                "manifest": m, "entry": rel["files"].get(m["models"][name]["file"])}
    raise ModelUnavailable(f"no compatible release of the {name!r} model "
                           f"(this code reads {compatibility()})"
                           + (f"; version {version!r} not found" if version else ""))


def fetch(name: str = "8d", *, version: Optional[str] = None) -> Path:
    """The local path of model ``name``, downloading it into the cache if needed."""
    rel = resolve(name, version=version)
    info = rel["manifest"]["models"][name]
    path = cache_dir() / str(rel["record"]) / info["file"]
    if _cached_ok(path, info):
        return path
    if path.exists():                       # corrupt or truncated: never return it
        path.unlink()
    if rel["entry"] is None:
        raise ModelUnavailable(f"{path} is not cached and LMID_OFFLINE is set")
    return _download(rel["entry"], path, sha256=info.get("sha256"))


def problem(model):
    """The spectral grid ``model`` was built on (a ``solver_3d.Problem3D``), for
    ``evaluate_polished`` and ``solver_3d.evaluate_field``."""
    from .solver import solver_3d
    from .solver.operators_abt import scheme_of
    meta = model.meta
    bmap, radial = str(meta.get("bmap", "cos")), str(meta.get("radial", "lobatto"))
    name = scheme_of(bmap, radial)
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    # Both branches name the grid as keywords, so tests/test_scheme_v2.py's source scan
    # can see it (a **kwargs call cannot be verified there and is reported).
    if name is not None:
        return solver_3d.make_problem(Na, Nb, Nphi, scheme=name)
    return solver_3d.make_problem(Na, Nb, Nphi, bmap=bmap, radial=radial)


def load(name: str = "8d", *, version: Optional[str] = None, path=None,
         solver: bool = False, **attach_kw):
    """Load a shipped model.

    The returned model's ``evaluate(theta)`` is the interpolated field on the grid
    nodes (numpy), ``evaluate_jax(theta)`` its differentiable twin, and, with
    ``solver=True``, ``evaluate_polished(theta)`` the certified field.

    Parameters
    ----------
    name : ``"4d"`` or ``"8d"``
        The aligned-spin model θ = (b, q, χ_Ay, χ_By) or the general-spin one
        θ = (b, q, χ_A, χ_B).  ``model.meta["axis_names"]`` lists the order.
    version : str, optional
        A Zenodo version to pin; default the newest one this code can read.
    path : path-like, optional
        A local ``.npz`` instead of a download.
    solver : bool
        Attach the Newton–Krylov solver on the model's own grid, so that
        ``model.evaluate_polished(theta)`` returns certified data.  It builds the
        grid's operators once (a few seconds).
    **attach_kw
        Passed to ``parametric_nd.attach_solve_fn_3d`` (``gmres_rtol``,
        ``retry_tol``, ...).
    """
    from .parametric.hermite_smolyak_pod_cross import load_pod_hermite_smolyak_cross
    model = load_pod_hermite_smolyak_cross(str(path) if path is not None
                                           else fetch(name, version=version))
    if solver:
        from .parametric.parametric_nd import attach_solve_fn_3d
        meta = model.meta
        attach_solve_fn_3d(model, problem(model), list(meta["axis_names"]),
                           M_tot=float(meta.get("M_tot", 1.0)),
                           fixed=meta.get("fixed") or None, **attach_kw)
    return model
