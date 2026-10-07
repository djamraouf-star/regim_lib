"""Manifeste de l'étude effectuée."""
import pandas as pd
from regime_lib.utils.provenance import json_value, file_hash, frame_info, environment

def manifest(study, *, origin):
    config = {name: getattr(study, name) for name in (
        'noms_features', 'noms_cibles', 'methodes', 'methodes_causales_only',
        'schema_split', 'split_kwargs', 'feature_types', 'tests_modalite', 'stratify_by',
        'asset', 'timeframe', 'configurations', 'ohlcv_asset', 'ohlcv_timeframe')}
    config['inference'] = study._inference_metadata()
    mode = 'exploration_in_sample' if study.schema_split == 'in_sample' else 'evaluation_temporelle'
    return json_value({
        'schema_version': 1, 'origin': origin, 'analysis_mode': mode,
        'validation_claim': 'Aucune validation indépendante finale certifiée automatiquement.',
        'configuration': config, **environment(),
        'sources': getattr(study, '_source_metadata', {}),
        'features': frame_info(study.df_features), 'targets': frame_info(study.df_cibles),
        'strata': frame_info(study.df_strates),
        'stratified_results': frame_info(study.resultats_stratifies),
        'strata_coverage': frame_info(study.couverture_strates),
        'target_ends': frame_info(study.df_fin_cibles),
        'expected_index': frame_info(pd.DataFrame(index=study.expected_index)) if study.expected_index is not None else None,
        'folds': getattr(study, '_fold_metadata', []),
        'results': frame_info(study.resultats),
        'coverage': study.couverture.to_dict('records') if study.couverture is not None else [],
        'common_support': frame_info(study.support_commun),
    })
