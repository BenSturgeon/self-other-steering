# Introspection Evals

Experiments testing self-prediction and introspection in language models, inspired by ["Looking Inward: Language Models Can Learn About Themselves by Introspection"](https://arxiv.org/abs/2410.13787).

## Goal

Test whether CAA steering on the self-other axis can **improve** a model's ability to accurately predict its own behavior - enhancing introspection by amplifying self-modeling circuits.

## Structure

```
introspection-evals/
├── data/           # Datasets for self-prediction tasks
├── evals/          # Evaluation implementations
├── scripts/        # Run experiments
├── configs/        # Experiment configurations
└── results/        # Output and analysis
```

## Key Experiments

1. **Self-prediction baseline**: Can Qwen3-32B predict its own responses to hypothetical inputs?
2. **Steering impact**: Does CAA on self-other axis improve self-prediction accuracy?
3. **Control comparisons**: Do random/control vectors have similar effects?

## Setup

```bash
# From self_modelling_steering root
source .venv/bin/activate
cd introspection-evals
```

## References

- [Looking Inward paper](https://arxiv.org/abs/2410.13787)
- [Original code](https://github.com/felixbinder/introspection_self_prediction)
