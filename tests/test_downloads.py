import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('encoder_download', ROOT / 'scripts/download_encoders.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def test_verified_download_and_conflicting_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.write_bytes(b'fixed test bytes')
            entry = {'url': source.as_uri(), 'bytes': 16,
                     'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            target = root / 'output' / 'weight'
            module.download(entry, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            source.unlink()
            module.download(entry, target)
            target.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                module.download(entry, target)

    def test_invalid_download_never_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.write_bytes(b'corrupt')
            target = root / 'output'
            with self.assertRaises(ValueError):
                module.download({'url': source.as_uri(), 'bytes': 7,
                                 'sha256': '0' * 64}, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob('*.part')), [])


if __name__ == '__main__':
    unittest.main()
