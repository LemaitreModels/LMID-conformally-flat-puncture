"""The model download (``models.py``) against a fake Zenodo on localhost.

What must hold, because each failure would be silent for a user:

* the newest *compatible* version is chosen, and a newer incompatible one is
  skipped with a warning rather than loaded into code that cannot read it;
* a pinned version is honoured, and an incompatible pinned version refuses;
* a corrupted download is rejected and never left in the cache;
* a cached file is reused without downloading it again, a corrupted cached file is
  replaced, and ``LMID_OFFLINE`` works from the cache alone;
* ``load(path=...)`` and ``problem(model)`` give a model and its grid.

No network: a ``http.server`` thread plays Zenodo's records API.
"""
import hashlib
import http.server
import json
import threading

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture import models as M


def _manifest(version, payload, **override):
    m = {"version": version, **M.compatibility(), "models": {
        "4d": {"file": f"model-{version}.npz", "size": len(payload),
               "sha256": hashlib.sha256(payload).hexdigest()}}}
    m.update(override)
    return json.dumps(m).encode()


class _FakeZenodo:
    """Concept 100 with versions 101 (1.0.0), 102 (1.1.0) and 103 (2.0.0, which needs
    a newer model format).  Record 102's model file can be served corrupted."""

    def __init__(self):
        self.files, self.records, self.hits = {}, [], []
        for rid, ver, fmt in ((101, "1.0.0", M.MODEL_FORMAT), (102, "1.1.0", M.MODEL_FORMAT),
                              (103, "2.0.0", M.MODEL_FORMAT + 1)):
            payload = f"model bytes of {ver}".encode()
            man = _manifest(ver, payload, model_format=fmt)
            self.files[rid] = {"manifest.json": man, f"model-{ver}.npz": payload}
            self.records.append((rid, ver))
        self.corrupt = False

    def record(self, rid, base):
        ver = dict(self.records)[rid]
        files = [{"key": k, "size": len(v), "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                  "links": {"self": f"{base}/records/{rid}/files/{k}/content"}}
                 for k, v in self.files[rid].items()]
        return {"id": rid, "doi": f"10.5281/zenodo.{rid}", "files": files,
                "metadata": {"version": ver, "publication_date": "2026-11-12"}}

    def handler(self):
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.hits.append(self.path)
                base = f"http://127.0.0.1:{self.server.server_port}/api"
                parts = self.path.split("?")[0].strip("/").split("/")
                if parts[:2] == ["api", "records"] and len(parts) == 3:
                    rid = int(parts[2])
                    if rid == 100:                       # the concept: redirect to newest
                        self.send_response(302)
                        self.send_header("Location", f"{base}/records/103")
                        self.end_headers()
                        return
                    return self._send(200, json.dumps(fake.record(rid, base)).encode())
                if parts[:2] == ["api", "records"] and parts[3:] == ["versions"]:
                    hits = [fake.record(r, base) for r, _ in sorted(fake.records, reverse=True)]
                    return self._send(200, json.dumps({"hits": {"hits": hits}}).encode())
                if parts[:2] == ["api", "records"] and parts[3] == "files":
                    body = fake.files[int(parts[2])][parts[4]]
                    if fake.corrupt and parts[4].endswith(".npz"):
                        body = body[:-1] + b"X"
                    return self._send(200, body, "application/octet-stream")
                self._send(404, b"{}")
        return H


@pytest.fixture
def zenodo(tmp_path, monkeypatch):
    fake = _FakeZenodo()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LMID_ZENODO_API", f"http://127.0.0.1:{srv.server_port}/api")
    monkeypatch.setenv("LMID_ZENODO_RECORD", "100")
    monkeypatch.setenv("LMID_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("LMID_OFFLINE", raising=False)
    yield fake
    srv.shutdown()


def test_newest_compatible_version_is_chosen(zenodo):
    with pytest.warns(UserWarning, match="2.0.0"):
        rel = M.resolve("4d")
    assert rel["version"] == "1.1.0" and rel["record"] == 102


def test_fetch_downloads_checks_and_caches(zenodo):
    with pytest.warns(UserWarning):
        p = M.fetch("4d")
    assert p.read_bytes() == b"model bytes of 1.1.0"
    n = len([h for h in zenodo.hits if h.endswith(".npz/content")])
    with pytest.warns(UserWarning):
        assert M.fetch("4d") == p                  # second call: no new download
    assert len([h for h in zenodo.hits if h.endswith(".npz/content")]) == n


def test_pinned_version(zenodo):
    assert M.fetch("4d", version="1.0.0").read_bytes() == b"model bytes of 1.0.0"
    with pytest.raises(M.ModelUnavailable, match="needs"):
        M.fetch("4d", version="2.0.0")


def test_corrupt_download_is_rejected_and_not_cached(zenodo):
    zenodo.corrupt = True
    with pytest.warns(UserWarning), pytest.raises(M.ModelUnavailable, match="checksum"):
        M.fetch("4d")
    cached = list(M.cache_dir().rglob("*.npz")) + list(M.cache_dir().rglob("*.part"))
    assert cached == []


def test_offline_uses_the_cache(zenodo, monkeypatch):
    with pytest.warns(UserWarning):
        p = M.fetch("4d")
    monkeypatch.setenv("LMID_OFFLINE", "1")
    monkeypatch.setenv("LMID_ZENODO_API", "http://127.0.0.1:9/api")   # nothing listens
    assert M.fetch("4d") == p


def test_corrupt_cache_is_replaced(zenodo):
    with pytest.warns(UserWarning):
        p = M.fetch("4d")
    p.write_bytes(b"garbage of the same length!!")
    with pytest.warns(UserWarning):
        assert M.fetch("4d").read_bytes() == b"model bytes of 1.1.0"


def test_offline_skips_a_manifest_whose_file_is_missing(zenodo, monkeypatch):
    """A newer cached manifest without its model must not shadow an older release
    whose model is cached."""
    old = M.fetch("4d", version="1.0.0")
    with pytest.warns(UserWarning):
        M.resolve("4d")                    # caches the 1.1.0 manifest, not its model
    monkeypatch.setenv("LMID_OFFLINE", "1")
    assert M.fetch("4d") == old


def test_unpublished_record_says_so(monkeypatch):
    monkeypatch.delenv("LMID_ZENODO_RECORD", raising=False)
    monkeypatch.delenv("LMID_OFFLINE", raising=False)
    monkeypatch.setattr(M, "ZENODO_CONCEPT_RECORD", None)
    with pytest.raises(M.ModelUnavailable, match="not been published"):
        M.releases()


def test_load_local_file_and_its_grid(tmp_path):
    """``load(path=)`` on a tiny saved POD cross model, and ``problem`` rebuilds the
    grid its metadata names."""
    from lemaitre.initial_data.conformally_flat_puncture.parametric import (
        hermite_smolyak as hsm, hermite_smolyak_cross as hsc, hermite_smolyak_pod_cross as hspc)
    axes, enh, shape = [(3.0, 5.0), (-0.5, 0.5)], (1,), (7, 6, 2)

    class Info:
        iters, residual_norm, converged = 1, 1e-14, True

    def field(th):
        b, c = th
        return np.fromfunction(lambda i, j, k: np.sin(b + 0.1 * i) * (1 + c * j) + k, shape)

    def solve(th, guess, tol, it):
        return field(np.asarray(th, float)), Info()

    def tangent(th, U):
        th = np.asarray(th, float)
        return np.stack([(field(th + h) - field(th - h)) / 2e-6
                         for h in (np.array([1e-6, 0]), np.array([0, 1e-6]))])

    herm = hsm.HermiteSmolyakSolverND(solve, axes, tangent, enhanced_axes=enh).build_isotropic(2)
    pool = {k: (th, U, dU, np.zeros((0,) + shape), it, rs)
            for k, (th, U, dU, it, rs) in herm._dedup_pool().items()}
    cross = hsc.build_cross_from_pool(axes, herm.index_set, enh, pool)
    pod, _ = hspc.build_pod_hermite_smolyak_cross(cross, r=3)
    path = tmp_path / "tiny.npz"
    pod.save(path, meta={"axis_names": ["b", "chi_Ay"], "Na": 6, "Nb": 6, "Nphi": 2,
                         "bmap": "abt", "radial": "gauss"})
    model = M.load(path=path)
    th = np.array([3.7, 0.2])
    assert np.allclose(model.evaluate(th), pod.evaluate(th), rtol=0, atol=1e-12)
    prob = M.problem(model)
    assert (prob.Na + 1, prob.Nb, prob.Nphi) == shape
