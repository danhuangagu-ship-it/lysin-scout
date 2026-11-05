import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from lysin_scout.predict import Predictor, classify
from lysin_scout.embed import read_fasta
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_frozen_model_predictions(self):
        for name in ('ESMC', 'ProtBERT', 'ProSE'):
            with self.subTest(model=name):
                folder = ROOT / 'examples' / name
                ids = pd.read_csv(folder / 'ids.csv', dtype=str)
                expected = pd.read_csv(folder / 'expected_predictions.csv', float_precision='round_trip')
                self.assertEqual(ids.id.tolist(), expected.id.tolist())
                predictor = Predictor(name)
                features = np.load(folder / 'features.npy', allow_pickle=False)
                actual = predictor.probabilities(features, batch_size=2)
                ref = expected[[f'p_class_{i}' for i in range(4)]].to_numpy()
                np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-5)
                np.testing.assert_array_equal(actual.argmax(1), expected.predicted_class)
                result = classify(ids, actual, name)
                np.testing.assert_array_equal(result.is_candidate, actual.argmax(1) == 0)

    def test_invalid_features_rejected(self):
        predictor = Predictor('ESMC')
        for x in [np.zeros((2, 961)), np.full((1, 960), np.nan), np.empty((0, 960))]:
            with self.assertRaises(ValueError):
                predictor.probabilities(x)

    def test_duplicate_fasta_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.faa'
            p.write_text('>a\nMAGX*\n>a\nMALG\n')
            with self.assertRaises(ValueError):
                read_fasta(p)


if __name__ == '__main__':
    unittest.main()
