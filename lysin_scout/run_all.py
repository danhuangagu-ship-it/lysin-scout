"""Run the three unchanged predictors sequentially and join results by protein ID."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .embed import read_fasta
from .predict import MODELS, Predictor, classify


def aligned_ids(tables, expected=None):
    """Require exactly the same IDs and raw sequence hashes, regardless of row order."""
    canonical = expected
    for name in MODELS:
        ids = tables[name]
        if not {'id', 'sequence_sha256'}.issubset(ids.columns):
            raise ValueError(f'{name}: missing id/sequence_sha256 columns')
        if ids.id.eq('').any() or not ids.id.is_unique or not ids.sequence_sha256.str.fullmatch('[0-9a-f]{64}').all():
            raise ValueError(f'{name}: invalid IDs or sequence hashes')
        current = dict(zip(ids.id, ids.sequence_sha256))
        if canonical is None:
            canonical = current
        elif current != canonical:
            raise ValueError(f'{name}: protein IDs or raw sequence hashes differ between model inputs')
    return canonical


def summarize(results):
    tables = {name: results[name][['id', 'sequence_sha256']] for name in MODELS}
    canonical = aligned_ids(tables)
    summary = pd.DataFrame({'id': list(canonical), 'sequence_sha256': list(canonical.values())})
    for name in MODELS:
        data = results[name].set_index('id').loc[summary.id]
        for field in ('predicted_class', 'predicted_label', 'endolysin_score', 'is_candidate',
                      'passes_recommended_threshold'):
            summary[f'{name}_{field}'] = data[field].to_numpy()
    summary['supporting_models'] = sum(summary[f'{name}_is_candidate'] for name in MODELS)
    summary['models_passing_threshold'] = sum(summary[f'{name}_passes_recommended_threshold'] for name in MODELS)
    summary['candidate_any_model'] = summary.supporting_models.gt(0).astype(int)
    summary['candidate_all_models'] = summary.supporting_models.eq(3).astype(int)
    return summary


def run_pipeline(args):
    models = MODELS if args.model == "all" else (args.model,)
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError('Output directory exists; choose a new path')
    tables, folders, commands = {}, {}, {}
    expected = None
    if args.fasta:
        records = read_fasta(args.fasta)
        expected = {i: hashlib.sha256(s.encode()).hexdigest() for i, s in records}
        config_path = Path(args.encoder_config).resolve()
        config = json.loads(config_path.read_text())
        if not set(models).issubset(config):
            raise ValueError('Encoder config is missing a requested model')
        for name in models:
            spec = config[name]
            required = {'python', 'encoder_path'} | ({'prose_source'} if name == 'ProSE' else set())
            if not required.issubset(spec) or set(spec) - {'python', 'encoder_path', 'prose_source'}:
                raise ValueError(f'{name}: invalid encoder config fields')
            paths = {k: (config_path.parent / spec[k]).absolute() for k in required}
            for key, value in paths.items():
                if not value.exists():
                    raise ValueError(f'{name}: {key} not found: {value}')
            folder = out.resolve() / 'embeddings' / name
            commands[name] = [str(paths['python']), '-m', 'lysin_scout.embed', '--model', name,
                              '--fasta', str(Path(args.fasta).resolve()), '--encoder-path', str(paths['encoder_path']),
                              '--out-dir', str(folder), '--device', args.device, '--threads', str(args.threads)]
            if name == 'ProSE':
                commands[name] += ['--prose-source', str(paths['prose_source'])]
            folders[name] = folder
    else:
        for name in models:
            folder = Path(args.features_dir) / name
            if not (folder / 'features.npy').is_file():
                raise ValueError(f'{name}: features.npy not found')
            folders[name] = folder
            tables[name] = pd.read_csv(folder / 'ids.csv', dtype=str, keep_default_na=False)
        aligned_ids(tables)
    out.mkdir(parents=True)
    manifest = {'status': 'running', 'input_mode': 'fasta' if args.fasta else 'embeddings',
                'models': list(models), 'completed_models': [],
                'device': args.device if args.fasta else 'cpu', 'threads': args.threads,
                'score_aggregation': 'none; model scores are reported separately'}
    manifest_path = out / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = {}
    try:
        for name in models:
            manifest['current_model'] = name
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if args.fasta:
                print(f'[{name}] Generating embeddings in its encoder environment', flush=True)
                subprocess.run(commands[name], check=True)
                tables[name] = pd.read_csv(folders[name] / 'ids.csv', dtype=str, keep_default_na=False)
                # Verify every generated input against the original full FASTA.
                ids = tables[name]
                if not {'id', 'sequence_sha256'}.issubset(ids) or not ids.id.is_unique:
                    raise ValueError(f'{name}: invalid generated IDs')
                if dict(zip(ids.id, ids.sequence_sha256)) != expected:
                    raise ValueError(f'{name}: generated IDs/hashes differ from FASTA')
            print(f'[{name}] Classifying', flush=True)
            features = np.load(folders[name] / 'features.npy', allow_pickle=False, mmap_mode='r')
            predictor = Predictor(name, threads=args.threads)
            results[name] = classify(tables[name], predictor.probabilities(features, args.batch_size), name)
            results[name].to_csv(out / f'{name}_predictions.csv', index=False, float_format='%.17g')
            manifest['completed_models'].append(name)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        if len(models) == 1:
            manifest.update(status="completed", proteins=len(results[models[0]]))
            manifest.pop("current_model", None)
            return
        summary = summarize(results)
        summary.to_csv(out / 'summary.csv', index=False, float_format='%.17g')
        manifest.update(status='completed', proteins=len(summary),
                        candidates_any_model=int(summary.candidate_any_model.sum()),
                        candidates_all_models=int(summary.candidate_all_models.sum()))
        manifest.pop('current_model', None)
        print(f'{len(summary)} proteins; {manifest["candidates_any_model"]} candidates from at least one model')
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


def run_all(args):
    args.model = "all"
    run_pipeline(args)
