"""Empreintes et environnement des calculs de détection et d'étude."""
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd

PROTOCOL_VERSION = "1.0.0"

def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_info(frame):
    if frame is None:
        return None
    digest = sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(repr([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    digest.update(json.dumps(json_value(frame.attrs), sort_keys=True, default=str).encode())
    return {'rows': len(frame), 'columns': list(frame.columns),
            'start': str(frame.index.min()) if len(frame) else None,
            'end': str(frame.index.max()) if len(frame) else None,
            'sha256': digest.hexdigest(), 'attrs': json_value(frame.attrs)}


def environment():
    versions = {"python": platform.python_version()}
    for package in ("regime_lib", "numpy", "pandas", "scipy", "pyarrow", "hmmlearn", "ruptures", "pyyaml"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    root = Path(__file__).parents[1]
    return {"protocol_version": PROTOCOL_VERSION, "versions": versions,
            "code_sha256": {str(p.relative_to(root)): file_hash(p)
                            for p in sorted(root.rglob("*.py"))},
            "resources_sha256": {str(p.relative_to(root)): file_hash(p)
                                 for pattern in ("*.yaml", "*.csv")
                                 for p in sorted(root.rglob(pattern))}}
