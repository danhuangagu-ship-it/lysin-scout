# LysinScout

PLM-guided discovery of phage endolysins

Antimicrobial resistance motivates the search for new antibacterial strategies.
Phage endolysins degrade bacterial peptidoglycan and are being investigated as
antibacterial agents. Their sequence diversity can make discovery through
conventional homology-based annotation challenging.

LysinScout uses three protein-language-model predictors—ESMC-MLP, ProtBERT-MLP and
ProSE-MLP—to screen phage proteins for candidate endolysins. The predictors
distinguish endolysins, virion-associated lysins (VALs), holins and other viral
proteins (Non-EVH), supporting candidate selection for further functional testing.

**Input:** a FASTA file containing amino-acid sequences of proteins predicted from
phage genomes or contigs. Each protein must have a unique FASTA identifier.

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

deactivate
```

### Screen a phage protein FASTA

Set up the ESMC encoder environment and download its weights using the
[installation guide](docs/INSTALL.md). Adjust the ESMC paths in
`encoders.example.json` if needed. Then run:

```bash
source .venv-classifier/bin/activate
lysin-scout --model ESMC --fasta phage_proteins.faa \
  --encoder-config encoders.example.json --out-dir results/esmc --device cpu
```

LysinScout generates embeddings in the configured encoder environment and writes
the predictions to `results/esmc/ESMC_predictions.csv`. For ProtBERT or ProSE,
set up the corresponding encoder and change the model name.

### Run all three models

For the included embeddings, run all three classifiers in one command:

```bash
source .venv-classifier/bin/activate
lysin-scout --model all --features-dir examples --out-dir results/all_examples
```

To start from your **phage protein FASTA**, first install all three encoder
environments and download their weights as described in [the installation guide](docs/INSTALL.md).
Edit `encoders.example.json` to point to the corresponding Python executables,
weights and ProSE source directory. Relative paths are resolved from the directory
containing that JSON file. Then run:

```bash
source .venv-classifier/bin/activate
lysin-scout --model all --fasta phage_proteins.faa \
  --encoder-config encoders.example.json --out-dir results/all_models --device cpu
```

The runner sequentially generates embeddings in each encoder's own environment
and classifies them in the current classifier environment. It writes
`ESMC_predictions.csv`, `ProtBERT_predictions.csv`, `ProSE_predictions.csv` and
`summary.csv`. The summary preserves individual scores, classes and threshold
flags, and reports support by any or all models. Scores are not averaged across
models. IDs and raw sequence hashes must match across the three inputs; input
row order can differ. `run_manifest.json` records completion or failure, and the
summary is produced only when all three models finish. Use a new output directory
for each run. Model-specific end-to-end validation limitations below still apply.

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
