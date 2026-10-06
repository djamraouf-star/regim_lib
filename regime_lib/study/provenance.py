"""Manifestes reproductibles des données et paramètres effectivement étudiés."""
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


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


def manifest(study, *, origin):
    versions = {'python': platform.python_version()}
    for package in ('regime_lib', 'numpy', 'pandas', 'scipy', 'pyarrow'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'unavailable'
    config = {name: getattr(study, name) for name in (
        'noms_features', 'noms_cibles', 'methodes', 'methodes_causales_only',
        'schema_split', 'split_kwargs', 'feature_types', 'tests_modalite',
        'asset', 'timeframe', 'configurations', 'ohlcv_asset', 'ohlcv_timeframe')}
    config['inference'] = study.inference.to_dict() if study.inference else None
    mode = 'exploration_in_sample' if study.schema_split == 'in_sample' else 'evaluation_temporelle'
    return json_value({
        'schema_version': 1, 'origin': origin, 'analysis_mode': mode,
        'validation_claim': 'Aucune validation indépendante finale certifiée automatiquement.',
        'configuration': config, 'versions': versions,
        'sources': getattr(study, '_source_metadata', {}),
        'features': frame_info(study.df_features), 'targets': frame_info(study.df_cibles),
        'target_ends': frame_info(study.df_fin_cibles),
        'expected_index': frame_info(pd.DataFrame(index=study.expected_index)) if study.expected_index is not None else None,
        'folds': getattr(study, '_fold_metadata', []),
        'results': frame_info(study.resultats),
        'coverage': study.couverture.to_dict('records') if study.couverture is not None else [],
        'code_sha256': {str(p.relative_to(Path(__file__).parents[1])): file_hash(p)
                        for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
    })
