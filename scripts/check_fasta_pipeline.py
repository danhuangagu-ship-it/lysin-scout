"""Optional real-encoder smoke test. Downloads are never triggered by this test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIMS = {'ESMC': 960, 'ProtBERT': 1024, 'ProSE': 6165}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=DIMS, required=True)
    parser.add_argument('--encoder-path', required=True)
    parser.add_argument('--prose-source')
    parser.add_argument('--classifier-python', default=sys.executable)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='lysin-smoke-') as tmp:
        out = Path(tmp) / 'embeddings'
        cmd = [sys.executable, '-m', 'lysin_scout.embed', '--model', args.model,
               '--fasta', str(ROOT / 'tests/integration/smoke.faa'),
               '--encoder-path', args.encoder_path, '--out-dir', str(out),
               '--device', args.device, '--threads', '1']
        if args.prose_source:
            cmd += ['--prose-source', args.prose_source]
        subprocess.run(cmd, check=True, cwd=ROOT)
        features = np.load(out / 'features.npy', allow_pickle=False)
        assert features.shape == (3, DIMS[args.model]), features.shape
        assert np.isfinite(features).all()
        np.testing.assert_array_equal(features[0], features[1])
        with (out / 'ids.csv').open() as f:
            ids = list(csv.DictReader(f))
        assert [x['id'] for x in ids] == ['example_a', 'example_duplicate', 'example_stop']
        base = 'MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQ'
        for record, sequence in zip(ids, [base, base, base + '*']):
            effective = sequence.rstrip('*') if args.model == 'ProtBERT' else sequence
            assert record['sequence_sha256'] == hashlib.sha256(sequence.encode()).hexdigest()
            assert record['encoded_sequence_sha256'] == hashlib.sha256(effective.encode()).hexdigest()
        if args.model == 'ProtBERT':
            np.testing.assert_array_equal(features[0], features[2])
        predictions = Path(tmp) / 'predictions.csv'
        subprocess.run([args.classifier_python, '-m', 'lysin_scout.predict',
                        '--model', args.model, '--features', str(out / 'features.npy'),
                        '--ids', str(out / 'ids.csv'), '--out', str(predictions)], check=True, cwd=ROOT)
        with predictions.open() as f:
            rows = list(csv.DictReader(f))
        assert [r['id'] for r in rows] == [r['id'] for r in ids]
        probs = np.array([[float(r[f'p_class_{i}']) for i in range(4)] for r in rows])
        assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        np.testing.assert_allclose(probs.sum(1), 1, rtol=0, atol=1e-6)
        assert [int(r['predicted_class']) for r in rows] == probs.argmax(1).tolist()
        result = json.loads((out / 'manifest.json').read_text())
        print(json.dumps({'smoke_test': 'passed', 'synthetic_sequences': True,
                          'classifier_python': args.classifier_python,
                          'encoder_manifest': result}, indent=2))


if __name__ == '__main__':
    main()
