# LysinScout

PLM-guided discovery of phage endolysins

LysinScout provides three trained four-class predictors: ESMC-MLP,
ProtBERT-MLP and ProSE-MLP. Each classifier distinguishes endolysin, virion-associated
lysin (VAL), holin and other viral proteins (Non-EVH).

The repository includes the original seed-42 MLP weights, associated training-set
scalers, configuration files and fixed embedding examples. Weights and scalers
are verified by SHA-256 before inference. These models are not retrained when
screening a new protein collection.

## Install

Use Python 3.10–3.12 in a separate environment:

```bash
git clone https://github.com/danhuangagu-ship-it/lysin-scout.git
cd lysin-scout
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The classifier examples were verified on CPU with Python 3.12, NumPy 1.26.4,
PyTorch 2.6.0, scikit-learn 1.3.2 and joblib 1.4.2. The classifiers were originally
trained with PyTorch 2.1.0 and scikit-learn 1.3.2.

## Quick start with the included embeddings

No pretrained encoder downloads are needed for this example:

```bash
lysin-scout --model ESMC \
  --features examples/ESMC/features.npy \
  --ids examples/ESMC/ids.csv \
  --out results/esmc_predictions.csv

python -m unittest discover -s tests -v
```

Replace `ESMC` with `ProtBERT` or `ProSE` to use the other examples. Tests compare
all three models against archived predictions, requiring identical predicted
classes and maximum absolute probability differences no greater than 1e-5.

## Predict from protein sequences

Generate embeddings first, then run the corresponding classifier. Each encoder
should run in an environment compatible with its upstream implementation.
Pretrained encoder weights are not included in this repository.

```bash
# Example: ESMC encoder environment, with upstream esm installed
python -m lysin_scout.embed --model ESMC --fasta proteins.faa \
  --encoder-path pretrained/esmc_300m_2024_12_v0.pth \
  --out-dir embeddings/ESMC --device cuda

# Classifier environment
lysin-scout --model ESMC --features embeddings/ESMC/features.npy \
  --ids embeddings/ESMC/ids.csv --out results/esmc_predictions.csv
```

For ProtBERT, install the upstream `transformers` package in the encoder environment
and pass a local `Rostlab/prot_bert` directory to `--encoder-path`. This is
ProtBERT, not ProtBERT-BFD. For ProSE, obtain the upstream ProSE checkout and its
`prose_mt_3x1024.sav` weights, then use:

```bash
python -m lysin_scout.embed --model ProSE --fasta proteins.faa \
  --prose-source /path/to/prose \
  --encoder-path /path/to/prose/saved_models/prose_mt_3x1024.sav \
  --out-dir embeddings/ProSE --device cpu
```

Obtain encoders from their upstream sources and follow their licenses and access
conditions: [ESM](https://github.com/evolutionaryscale/esm),
[ESMC checkpoint used in this study](https://huggingface.co/biohub/esmc-300m-2024-12),
[ProtBERT](https://huggingface.co/Rostlab/prot_bert), and
[ProSE](https://github.com/tbepler/prose).

The embedding command validates encoder parameter fingerprints against those
used for the classifiers. A mismatch fails rather than silently substituting
different weights. Sequences are not truncated; overly long inputs or insufficient
memory cause an error. Protein IDs must be unique; distinct IDs with identical
sequences are retained, with shared embedding computation.

| Encoder | Features | Pooling and sequence handling |
|---|---:|---|
| ESMC-300M | 960 | Mean of all returned tokens; terminal stop symbols retained |
| ProtBERT | 1,024 | Attention-mask mean including non-padding special tokens; terminal stop symbols removed |
| ProSE-MT | 6,165 | Mean of transformed residue representations; original sequence passed through Uniprot21 |

Only the classifier path and fixed embedding examples have been executed in the
local release test. Full encoder inference requires the matching upstream
dependencies and large weights; it is not covered by the small example tests.

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
