# Robot assets (fetched, never committed)

Third-party robot models (URDF / MJCF / USD, meshes) are **not** stored in
this repository. Each vendor publishes under its own license, so we pin exact
upstream commits in [`registry.py`](registry.py) and download them on demand:

```bash
python assets/fetch.py              # everything
python assets/fetch.py agibot/x2    # one robot
```

Fetched files land in `assets/<vendor>/<model>/` together with the upstream
LICENSE text (`assets/<vendor>/LICENSE-<repo>.txt`), which is the
authoritative license for those files. Everything under `assets/` except
`registry.py`, `fetch.py` and this README is gitignored.
