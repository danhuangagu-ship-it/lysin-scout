import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from lysin_scout.predict import MODELS
from lysin_scout.run_all import aligned_ids, run_all

ROOT = Path(__file__).resolve().parents[1]


def args(out, **kwargs):
    values = dict(out_dir=str(out), features_dir=str(ROOT / 'examples'), fasta=None,
                  encoder_config=None, device='cpu', threads=1, batch_size=2)
    values.update(kwargs)
    return SimpleNamespace(**values)


class RunAllTests(unittest.TestCase):
    def test_three_real_classifiers_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'all'
            run_all(args(out))
            combined = pd.read_csv(out / 'summary.csv')
            support = np.zeros(len(combined), dtype=int)
            for name in MODELS:
                ref = pd.read_csv(ROOT / 'examples' / name / 'expected_predictions.csv').set_index('id').loc[combined.id]
                result = pd.read_csv(out / f'{name}_predictions.csv').set_index('id').loc[combined.id]
                np.testing.assert_allclose(result.endolysin_score, ref.p_class_0, rtol=0, atol=1e-5)
                np.testing.assert_array_equal(combined[f'{name}_predicted_class'], ref.predicted_class)
                support += ref.predicted_class.eq(0).to_numpy()
            np.testing.assert_array_equal(combined.supporting_models, support)
            np.testing.assert_array_equal(combined.candidate_any_model, support > 0)
            np.testing.assert_array_equal(combined.candidate_all_models, support == 3)
            self.assertEqual(json.loads((out / 'run_manifest.json').read_text())['status'], 'completed')
            with self.assertRaises(ValueError):
                run_all(args(out))

    def test_id_alignment_and_hash_conflicts(self):
        original = pd.read_csv(ROOT / 'examples/ESMC/ids.csv', dtype=str)
        tables = {n: original.iloc[::-1].copy() for n in MODELS}
        self.assertEqual(aligned_ids(tables), dict(zip(original.id, original.sequence_sha256)))
        tables['ProSE'].loc[tables['ProSE'].index[0], 'sequence_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            aligned_ids(tables)

    def test_fasta_orchestration_with_stub_encoders(self):
        # Exercises subprocess wiring only; real-encoder verification remains separate.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.faa'
            fasta.write_text('>first\nMAG\n>second\nMAG\n')
            weight = root / 'weights'
            weight.touch()
            config = {n: {'python': sys.executable, 'encoder_path': 'weights'} for n in MODELS}
            config['ProSE']['prose_source'] = '.'
            cfg = root / 'config.json'
            cfg.write_text(json.dumps(config))
            calls = []
            def encoder(command, check):
                name = command[command.index('--model') + 1]
                calls.append(name)
                out = Path(command[command.index('--out-dir') + 1])
                out.mkdir(parents=True)
                matrix = np.load(ROOT / 'examples' / name / 'features.npy')[:2]
                np.save(out / 'features.npy', matrix)
                pd.DataFrame({'id': ['first', 'second'],
                              'sequence_sha256': [hashlib.sha256(b'MAG').hexdigest()] * 2}).to_csv(out / 'ids.csv', index=False)
            out = root / 'results'
            with patch('lysin_scout.run_all.subprocess.run', side_effect=encoder):
                run_all(args(out, features_dir=None, fasta=str(fasta), encoder_config=str(cfg)))
            self.assertEqual(calls, list(MODELS))
            self.assertEqual(pd.read_csv(out / 'summary.csv').id.tolist(), ['first', 'second'])

    def test_encoder_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.faa'
            fasta.write_text('>first\nMAG\n')
            weight = root / 'weights'
            weight.touch()
            config = {n: {'python': sys.executable, 'encoder_path': 'weights'} for n in MODELS}
            config['ProSE']['prose_source'] = '.'
            cfg = root / 'config.json'
            cfg.write_text(json.dumps(config))
            out = root / 'results'
            with patch('lysin_scout.run_all.subprocess.run', side_effect=RuntimeError('encoder failed')):
                with self.assertRaises(RuntimeError):
                    run_all(args(out, features_dir=None, fasta=str(fasta), encoder_config=str(cfg)))
            manifest = json.loads((out / 'run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['current_model'], 'ESMC')
            self.assertFalse((out / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
