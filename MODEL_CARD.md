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
