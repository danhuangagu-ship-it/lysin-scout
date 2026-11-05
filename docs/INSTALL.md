# CPU installation, encoder downloads and optional integration tests

Run commands from the repository root. These recipes target **Linux x86_64 and
Python 3.11**. A classifier environment can also use Python 3.12 as tested by the
maintainers. Install Python through your institution's modules or conda before
creating a venv. Do not mix encoder environments. GPU environments are not yet
validated; substituting CUDA wheels is not part of this CPU recipe.

## 1. Classifier environment

```bash
python3.11 -m venv .venv-classifier
source .venv-classifier/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/classifier-cpu.txt
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
deactivate
```

## 2. Download fixed encoder files

The downloader uses only the Python standard library. It checks SHA-256 and byte
length before accepting a file, rejects a conflicting existing file and never
substitutes a different model. URLs, revisions and checksums are in
[`encoder-files.json`](../environments/encoder-files.json).

```bash
python3 scripts/download_encoders.py --model ESMC --out-dir pretrained
python3 scripts/download_encoders.py --model ProtBERT --out-dir pretrained
python3 scripts/download_encoders.py --model ProSE --out-dir pretrained
```

Download only the encoder you need. Required files:

| Encoder | Fixed source | Download size | Resulting path |
|---|---|---:|---|
| ESMC | biohub/esmc-300m-2024-12, revision `7f10b20ae75017b2dbc884070e03434515709a8d` | 1.33 GB | `pretrained/ESMC/esmc_300m_2024_12_v0.pth` |
| ProtBERT | Rostlab/prot_bert, revision `7a894481acdc12202f0a415dd567f6cfdb698908` | 1.68 GB plus <1 KB configuration/tokenizer files | `pretrained/ProtBERT/` |
| ProSE | Berger lab's `saved_models.zip`, pinned by content checksum | 582 MB archive; 391 MB additional space for the extracted multitask weight | `pretrained/ProSE/prose_mt_3x1024.sav` |

Sizes are decimal, rounded. The Hugging Face repositories were publicly readable
without authentication when inspected on 2026-11-05; future access conditions may
change. Follow upstream license terms, including the non-commercial ProSE license.
The ProSE server has no immutable revision API: the pinned archive checksum detects
changes and fails instead of accepting changed content. Downloading weights does
not by itself verify encoder output equivalence; runtime parameter fingerprints
are checked again before embedding.

## 3. ESMC CPU environment

**Proposed full-dependency recipe, pending maintainer runtime verification.** This
is deliberately separate from the user's successful selective PyTorch 2.6.0
installation. It pairs torchtext 0.18.0 and torchvision 0.18.0 with PyTorch 2.3.0.
All transitive package versions are pinned from a successful Linux/Python 3.11
CPU dependency resolution. It does not install ESM with `--no-deps`; httpx is explicitly included because the
ESM import graph can require it. Do not interpret a successful dependency resolver
as a passed `pip check` or a successful end-to-end run.

```bash
python3.11 -m venv .venv-esmc
source .venv-esmc/bin/activate
python -m pip install torch==2.3.0 torchvision==0.18.0 torchtext==0.18.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/esmc-cpu.txt
python -m pip install -e .
python -m pip check
python -m pip freeze > esmc-environment.txt

python -m lysin_scout.embed --model ESMC --fasta proteins.faa \
  --encoder-path pretrained/ESMC/esmc_300m_2024_12_v0.pth \
  --out-dir embeddings/ESMC --device cpu --threads 1

# Includes import, weight loading, embedding generation and the classifier.
python scripts/check_fasta_pipeline.py --model ESMC \
  --encoder-path pretrained/ESMC/esmc_300m_2024_12_v0.pth \
  --classifier-python "$PWD/.venv-classifier/bin/python" --device cpu
deactivate
```

ESM declares torchtext/torchvision as package dependencies even though this tool
only supports its ESMC sequence embedding path. They are included rather than
suppressing dependency checks. Other ESM functionality is outside this tool's test
scope. Do not call a selective installation dependency-complete if `pip check`
reports missing packages.

## 4. ProtBERT CPU environment

**Proposed recipe; end-to-end execution is not yet verified.**

```bash
python3.11 -m venv .venv-protbert
source .venv-protbert/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/protbert-cpu.txt
python -m pip install -e .
python -m pip check
python -m pip freeze > protbert-environment.txt
python -m lysin_scout.embed --model ProtBERT --fasta proteins.faa \
  --encoder-path pretrained/ProtBERT --out-dir embeddings/ProtBERT --device cpu
python scripts/check_fasta_pipeline.py --model ProtBERT \
  --encoder-path pretrained/ProtBERT \
  --classifier-python "$PWD/.venv-classifier/bin/python" --device cpu
deactivate
```

This is `Rostlab/prot_bert`, not ProtBERT-BFD. The downloader also fetches the
matching configuration and tokenizer. No online model lookup is used at inference.

## 5. ProSE CPU environment

**Linux recipe pending runtime verification.** The maintainers have passed the
synthetic FASTA smoke test on macOS ARM64 with Python 3.12.14, PyTorch 2.6.0 and
NumPy 1.26.4; this does not validate Linux installation. Download the weights
as above, and fetch the fixed upstream source revision:

```bash
mkdir -p upstream
git clone https://github.com/tbepler/prose.git upstream/prose
git -C upstream/prose checkout 3416cc8455438d154251cc57cd123e5e17807a7b
python3.11 -m venv .venv-prose
source .venv-prose/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/prose-cpu.txt
python -m pip install -e .
python -m pip check
python -m pip freeze > prose-environment.txt
python -m lysin_scout.embed --model ProSE --fasta proteins.faa \
  --prose-source upstream/prose --encoder-path pretrained/ProSE/prose_mt_3x1024.sav \
  --out-dir embeddings/ProSE --device cpu
python scripts/check_fasta_pipeline.py --model ProSE --prose-source upstream/prose \
  --encoder-path pretrained/ProSE/prose_mt_3x1024.sav \
  --classifier-python "$PWD/.venv-classifier/bin/python" --device cpu
deactivate
```

Only the upstream embedding modules are imported. The broader upstream training
and HDF5 workflows are not included in this dependency recipe.

## 6. Switch back to classification

```bash
source .venv-classifier/bin/activate
lysin-scout --model ESMC --features embeddings/ESMC/features.npy \
  --ids embeddings/ESMC/ids.csv --out results/esmc_predictions.csv
deactivate
```

Use the matching model, feature directory and IDs for ProtBERT or ProSE. A model's
feature dimension, preprocessing and encoder fingerprint must all agree.

## Run all three models together

After completing the three encoder installations and downloads, activate the
classifier environment. Update the paths in `encoders.example.json` (paths are
relative to that JSON file, or may be absolute), then run:

```bash
source .venv-classifier/bin/activate
lysin-scout --model all --fasta phage_proteins.faa \
  --encoder-config encoders.example.json --out-dir results/all_models --device cpu
deactivate
```

Each configured Python must have LysinScout installed with `python -m pip install
-e .` in its own environment as shown above. The runner performs ESMC, ProtBERT and
ProSE sequentially and stops on failure; completed per-model files remain available.
A failed run has `status: failed` in `run_manifest.json` and has no summary. Fix
the cause and choose a new output directory to rerun.

For precomputed embeddings, the layout is `ROOT/ESMC/{features.npy,ids.csv}`,
`ROOT/ProtBERT/{features.npy,ids.csv}` and `ROOT/ProSE/{features.npy,ids.csv}`:

```bash
source .venv-classifier/bin/activate
lysin-scout --model all --features-dir embeddings --out-dir results/all_from_embeddings
deactivate
```

`summary.csv` joins proteins by ID, checks agreement of raw sequence hashes and
retains each model's class, endolysin score, candidate flag and threshold flag.
`supporting_models` counts class-0 predictions (0–3); `models_passing_threshold`
counts models that also pass their own recommended threshold. `candidate_any_model`
and `candidate_all_models` denote union and three-model intersection, respectively.
These are support counts, not calibrated ensemble probabilities. Full three-encoder
FASTA execution still requires validation in the target environment; tests of the
runner with stored embeddings do not establish encoder compatibility.

## Optional tests and interpretation

`tests/integration/smoke.faa` contains synthetic sequences with no activity labels:
one sequence, an identical sequence under a different ID, and a terminal-stop
variant. `scripts/check_fasta_pipeline.py` checks the actual encoder's imports,
weight loading, finite features of the expected dimension, ID order, exact equality
of duplicate embeddings, raw/effective sequence hashes, ProtBERT terminal-stop
removal, and valid four-class probability outputs. ESMC/ProSE retain the terminal
stop in their effective input; their alphabets may map it to an unknown token.
The smoke test does not claim that synthetic sequences are endolysins, does not
measure predictive accuracy and does not establish all-hardware numerical agreement.
It is opt-in and never downloads multi-gigabyte encoders during unit tests.

## Resource measurement

No measured peak RAM, VRAM or FASTA runtime numbers were supplied in the user test,
so none are presented as benchmarks. Encoder file sizes are not peak-memory
estimates. CPU memory and runtime depend on protein length; attention memory can
increase substantially for long proteins. Batch size is one and no truncation is
performed.

On Linux with GNU time installed, measure a run after downloading the weights:

```bash
source .venv-esmc/bin/activate
/usr/bin/time -v -o esmc-resources.txt \
  python -m lysin_scout.embed --model ESMC --fasta tests/integration/smoke.faa \
  --encoder-path pretrained/ESMC/esmc_300m_2024_12_v0.pth \
  --out-dir embeddings/ESMC-resource-test --device cpu --threads 1
python -m pip freeze > esmc-packages.txt
lscpu > esmc-cpu.txt
deactivate
```

GNU time reports elapsed time and peak resident memory (kB on Linux). The embedding
manifest records Python/package versions, platform, threads and elapsed seconds
including encoder loading and fingerprint checks, but excluding initial Python/PyTorch
startup, file downloads and subsequent classification. GNU time measures the whole
encoder process.
For a separately validated CUDA environment, `--device cuda` additionally records
PyTorch peak allocated and reserved CUDA bytes. Those counters do not include all
memory used by the CUDA context or other processes; record GPU model/driver with
`nvidia-smi` separately. Use a new output directory for each run and retain input
lengths and the exact repository revision with the logs. Report classifier timing
separately if benchmarking the complete pipeline.
