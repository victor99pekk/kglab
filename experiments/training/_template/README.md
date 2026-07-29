# Training Experiment: <NNN> — <Title>

**Date:** YYYY-MM-DD
**Task:** entity_resolution | link_prediction | node_classification
**Model:** <model name / architecture>
**Status:** planned | running | completed

---

## Hypothesis

*What improvement do you expect from training this model? How will it affect KG quality?*

## Data

- **Training KG:** `<path to input KG JSON>`
- **Train/Val/Test split:** `<split ratios>`
- **Negative sampling strategy:** `<how negatives are generated>`

## Model

- **Architecture:** `<model description>`
- **Key hyperparameters:** `<epochs, lr, batch_size, embedding_dim>`

## Results

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1 | — | — | — |
| Wall Time | — | — | — |

*See `outputs/training_metrics.json` for the full report.*
*Best model checkpoint saved to `models/best.pt`.*

## Impact on KG Quality

*After applying the trained model to the pipeline, how did KG metrics change?*

| Before | After |
|---|---|
| ... | ... |

## Conclusions

*What did you learn? What should the next training experiment test?*
