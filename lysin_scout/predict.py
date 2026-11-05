"""Portable inference for the three published, unchanged MLP classifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('ESMC', 'ProtBERT', 'ProSE')
LABELS = ('endolysin', 'VAL', 'holin', 'Non-EVH')
THRESHOLDS = {'ESMC': 0.80, 'ProtBERT': 0.90, 'ProSE': 0.90}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Predictor:
    def __init__(self, name, threads=1, model_root=None):
        import torch
        import joblib
        if name not in MODELS or threads < 1:
            raise ValueError('Choose a supported model and a positive thread count')
        folder = Path(model_root or Path(__file__).parent / 'models') / (name + '_MLP')
        self.config = json.loads((folder / 'config.json').read_text())
        cfg = self.config
        for key in ('checkpoint', 'scaler'):
            if digest(folder / cfg[key]) != cfg[key + '_sha256']:
                raise ValueError('Model checksum mismatch: ' + key)
        torch.set_num_threads(threads)
        state = torch.load(folder / cfg['checkpoint'], map_location='cpu', weights_only=True)
        if state['input_dim'] != cfg['input_dim'] or state['params']['hidden_sizes'] != cfg['hidden_sizes']:
            raise ValueError('Checkpoint architecture differs from its configuration')
        layers = []
        width = cfg['input_dim']
        for hidden in cfg['hidden_sizes']:
            layers.extend([torch.nn.Linear(width, hidden), torch.nn.ReLU(), torch.nn.Dropout(cfg['dropout'])])
            width = hidden
        layers.append(torch.nn.Linear(width, 4))
        self.model = torch.nn.Sequential(*layers)
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.eval()
        self.scaler = joblib.load(folder / cfg['scaler'])
        self.name = name

    def probabilities(self, features, batch_size=512):
        import torch
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config['input_dim'] or not len(x) or not np.isfinite(x).all():
            raise ValueError('Feature dimension or values do not match this model')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        results = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                z = self.scaler.transform(x[start:start + batch_size]).astype(np.float32)
                results.append(self.model(torch.from_numpy(z)).softmax(1).numpy())
        return np.concatenate(results)


def classify(ids, probabilities, model):
    ids = ids.copy()
    if not {'id', 'sequence_sha256'}.issubset(ids) or ids['id'].isna().any() or not ids['id'].is_unique:
        raise ValueError('Unique IDs and row-matched sequence_sha256 values are required')
    if len(ids) != len(probabilities) or not ids['sequence_sha256'].astype(str).str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('IDs, sequence hashes and feature rows do not align')
    result = ids[['id', 'sequence_sha256']].copy()
    result['model'] = model
    result['predicted_class'] = probabilities.argmax(1)
    result['predicted_label'] = [LABELS[i] for i in result['predicted_class']]
    result['endolysin_score'] = probabilities[:, 0]
    result['is_candidate'] = result['predicted_class'].eq(0).astype(int)
    result['passes_recommended_threshold'] = (
        result['is_candidate'].eq(1) & result['endolysin_score'].gt(THRESHOLDS[model])).astype(int)
    for k in range(4):
        result[f'p_class_{k}'] = probabilities[:, k]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS + ('all',))
    parser.add_argument('--features', help='N x D .npy matrix from the matching encoder')
    parser.add_argument('--ids', help='Row-matched CSV: id,sequence_sha256')
    parser.add_argument('--out')
    parser.add_argument('--features-dir', help='For --model all: root containing MODEL/features.npy and ids.csv')
    parser.add_argument('--fasta', help='Phage protein FASTA input')
    parser.add_argument('--encoder-config', help='Encoder Python and weight paths for FASTA input')
    parser.add_argument('--out-dir', help='New output directory for FASTA or three-model runs')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='Encoder device for FASTA input')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error('--threads and --batch-size must be positive')
    if args.device != 'cpu' and not args.fasta:
        parser.error('--device cuda applies only to FASTA input; classifiers run on CPU')
    if args.fasta:
        if not args.out_dir or not args.encoder_config:
            parser.error('--fasta requires --encoder-config and --out-dir')
        if args.features or args.ids or args.out or args.features_dir:
            parser.error('Choose either FASTA or precomputed embeddings')
        from .run_all import run_pipeline
        run_pipeline(args)
        return
    if args.model == 'all':
        if not args.out_dir or bool(args.features_dir) == bool(args.fasta):
            parser.error('--model all requires --out-dir and exactly one of --features-dir or --fasta')
        if bool(args.encoder_config) != bool(args.fasta):
            parser.error('--encoder-config is required only with --fasta')
        if args.features or args.ids or args.out:
            parser.error('Use --features-dir/--fasta and --out-dir for --model all')
        from .run_all import run_all
        run_all(args)
        return
    if not all((args.features, args.ids, args.out)):
        parser.error('A single model requires --features, --ids and --out')
    if args.features_dir or args.fasta or args.encoder_config or args.out_dir:
        parser.error('Batch pipeline options require --model all')
    out = Path(args.out)
    if out.exists():
        parser.error('Output exists; choose a new path')
    ids = pd.read_csv(args.ids, dtype=str, keep_default_na=False)
    x = np.load(args.features, allow_pickle=False, mmap_mode='r')
    predictor = Predictor(args.model, args.threads)
    result = classify(ids, predictor.probabilities(x, args.batch_size), args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format='%.17g')
    print(f'{len(result)} proteins; {result.is_candidate.sum()} candidates')


if __name__ == '__main__':
    main()
