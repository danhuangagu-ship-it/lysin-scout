# Model card

## Intended use

Prioritization of phage protein candidates for endolysin research. The predictors
are four-class MLPs applied to fixed protein-language-model embeddings. They are
not antibacterial efficacy predictors and do not establish a protein's biological
role from a score alone.

## Training and evaluation

The fixed fitting and validation partitions contained 39,996 and 9,999 protein
records, respectively. The internal evaluation set contained 5,603 proteins. The
external panel contained 66 ProGen-associated lysozyme-positive sequences and
266 natural phage protein negatives. External lysozyme activity labels are not
direct evidence of a natural endolysin lifecycle role.

The three distributed classifiers use seed 42 and the minimum-validation-macro-
log-loss checkpoint. Each uses one 256-unit hidden layer, ReLU and dropout 0.3
during training; dropout is disabled during inference. StandardScaler is fitted
on the fitting partition and is included unchanged. Individual configuration files
record checkpoint hashes, dimensions, selection epochs and hyperparameters.

## Limitations

- The four softmax outputs are model scores, not calibrated functional probabilities.
- Threshold recommendations may remove divergent or functionally unassigned proteins.
- Catalytic domains can occur in VALs and other cell-wall-active proteins.
- A predicted holin neighbor is contextual computational evidence, not activity validation.
- Most discovered proteins have not been tested experimentally.
- Changing encoder weights, pooling, sequence handling or scaler invalidates equivalence
  to the provided classifier. Dimension agreement alone is insufficient.
- Encoder generation is adapted from the study's screening code. Full FASTA inference
  must be checked in an environment with matching upstream weights and dependencies.

## Local release checks

All three distributed models are compared against archived embedding examples.
The tests require class agreement and probability differences <=1e-5. They also
check rejection of invalid embeddings and duplicate FASTA IDs. These checks do
not measure generalization on a newly supplied collection.

## Encoder validation scope

A user reported a successful ESMC CPU FASTA pipeline on Linux HPC at commit
`ec0231d93162abae0b6d68facd5bd5e69a8489d2`, using esm 3.2.1,
transformers 4.48.1 and CPU PyTorch 2.6.0 after selectively installing dependencies
and adding httpx. Python version, full package inventory and hardware were not
provided. This report does not establish dependency completeness, ProtBERT/ProSE
end-to-end correctness or GPU support. The separately supplied full-dependency
ESMC recipe uses a different PyTorch version and needs its own runtime verification.
See the README validation table before choosing an environment.

A subsequent maintainer smoke test passed the ProSE FASTA-to-classifier path on
macOS ARM64, Python 3.12.14, PyTorch 2.6.0 and NumPy 1.26.4 with fixed upstream
source/weights and three synthetic records. This is a software integration check,
not a performance or activity benchmark, and does not validate the Linux recipe.
