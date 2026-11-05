# LysinScout

PLM-guided discovery of phage endolysins

LysinScout provides three trained four-class predictors: ESMC-MLP,
ProtBERT-MLP and ProSE-MLP. Each classifier distinguishes endolysin, virion-associated
lysin (VAL), holin and other viral proteins (Non-EVH).

The repository includes the original seed-42 MLP weights, associated training-set
scalers, configuration files and fixed embedding examples. Weights and scalers
are verified by SHA-256 before inference. These models are not retrained when
screening a new protein collection.

## Install and run

Use separate environments for classification and embedding generation. Do not run
an unversioned `pip install esm` in the classifier environment: upstream releases
may require a newer PyTorch and install large CUDA dependencies. The commands below
use Linux x86_64 CPU wheels and Python 3.11; see [installation details](docs/INSTALL.md)
for all encoders, fixed downloads, optional integration tests and GPU limitations.

```bash
git clone https://github.com/danhuangagu-ship-it/lysin-scout.git
cd lysin-scout
python3.11 -m venv .venv-classifier
source .venv-classifier/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/classifier-cpu.txt
python -m pip install -e .
python -m pip check

# No encoder download is needed for the archived-embedding example.
lysin-scout --model ESMC --features examples/ESMC/features.npy \
  --ids examples/ESMC/ids.csv --out results/esmc_example.csv
python -m unittest discover -s tests -v
deactivate
```

Replace `ESMC` with `ProtBERT` or `ProSE` in the example paths and model argument.
Each example has 16 row-matched embeddings and archived predictions.

### Validation status

| Path | Environment and evidence | Status |
|---|---|---|
| Three classifiers on archived embeddings | Maintainer: macOS ARM64, Python 3.12, PyTorch 2.6.0, NumPy 1.26.4, scikit-learn 1.3.2, joblib 1.4.2 | Tested; predicted classes agree, maximum probability difference about 1.2e-7 |
| Three classifiers on archived embeddings | User-reported Linux HPC test of `ec0231d`; exact Python/hardware not supplied | Reported passed; not independently reproduced |
| ESMC FASTA → embeddings → classifier, CPU | User-reported Linux HPC, esm 3.2.1, transformers 4.48.1, PyTorch 2.6.0; selective dependency installation plus httpx | Reported passed; not a dependency-complete validated environment |
| Full-dependency ESMC CPU recipe in `environments/` | Proposed Python 3.11 / PyTorch 2.3.0 stack matching torchtext 0.18 and torchvision 0.18 | Linux dependency resolution passed; runtime and `pip check` not yet verified |
| ProSE FASTA → embeddings → classifier, CPU | Maintainer: macOS ARM64, Python 3.12.14, PyTorch 2.6.0, NumPy 1.26.4, fixed upstream source/weights | Synthetic 3-record end-to-end smoke test passed; Linux recipe not runtime-verified |
| ProtBERT FASTA path | Proposed pinned core dependencies; optional test supplied | End-to-end execution not yet verified |
| GPU FASTA paths | No release benchmark supplied | Not yet verified |

Passing the classifier examples does not validate encoder imports, weights or FASTA
handling. Run the optional real-encoder test after installation. The ESMC Linux
file pins its full resolver-selected dependency set; the other recipes pin core
packages. Dependency resolution is not runtime validation. Keep the output of
`pip check` and `pip freeze` with your test report. Peak memory and end-to-end timings have not yet been reported;
[resource instructions](docs/INSTALL.md#resource-measurement) show how to collect them.

## Encoders and sequence handling

Pretrained encoders are downloaded separately with fixed revisions and file SHA-256
checksums; see `environments/encoder-files.json` and `scripts/download_encoders.py`.
The embedding command additionally checks parameter fingerprints against those used
to train the classifiers. Sequences are not truncated. IDs must be unique; identical
sequences under distinct IDs remain separate records and share embedding computation.

| Encoder | Features | Pooling and sequence handling |
|---|---:|---|
| ESMC-300M | 960 | Mean of all returned tokens; terminal stop symbols retained |
| ProtBERT | 1,024 | Attention-mask mean including non-padding special tokens; terminal stop symbols removed |
| ProSE-MT | 6,165 | Mean of transformed residue representations; original sequence passed through Uniprot21 |

Encoder licenses and access conditions remain those of their upstream providers:
[ESM](https://github.com/evolutionaryscale/esm),
[original-format ESMC weights](https://huggingface.co/biohub/esmc-300m-2024-12),
[ProtBERT](https://huggingface.co/Rostlab/prot_bert) and
[ProSE](https://github.com/tbepler/prose). Do not substitute a newer-format ESMC
checkpoint without verifying the parameter fingerprint and pooling conventions.

## Output and candidate selection

The CSV contains `id`, `sequence_sha256`, model name, predicted class and label,
endolysin score, candidate status, threshold flag and all four class probabilities.

| Class | Label |
|---:|---|
| 0 | endolysin |
| 1 | VAL |
| 2 | holin |
| 3 | Non-EVH |

`is_candidate = 1` means the four-class argmax is class 0. No additional score
cutoff changes this candidate label. The endolysin score is the class-0 probability.
`passes_recommended_threshold` is a separate prioritization flag, requiring both
class 0 and a strict score threshold:

| Model | Prioritization threshold |
|---|---:|
| ESMC-MLP | >0.80 |
| ProtBERT-MLP | >0.90 |
| ProSE-MLP | >0.90 |

These are retrospective prioritization settings, not calibrated guarantees of
endolysin function. Genome context and annotation review should complement
score filtering. This package does not itself predict structures, perform HMM
annotation or infer gene adjacency. Experimental activity requires validation.

## Files and provenance

`lysin_scout/models/` contains the unchanged trained classifiers and scalers.
`examples/` contains row-matched embeddings, sequence hashes and expected outputs.
`MODEL_CARD.md` describes scope and limitations. `SHA256SUMS` records distributable
file hashes. The encoder implementations and their large weights remain upstream.

Publication and licensing information will be added by the authors. No new
license grant is made here, and upstream encoder licenses remain applicable.
