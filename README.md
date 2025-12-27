# Introspection Evals

Experiments testing self-prediction and introspection in language models, inspired by ["Looking Inward: Language Models Can Learn About Themselves by Introspection"](https://arxiv.org/abs/2410.13787).

## Context

We found that CAA steering on the self-other axis toward "other" (negative α) causes models to:
- Use third-person framing ("The assistant feels...")
- Attribute richer experiences ("wonder", "awe", "curiosity")
- Bypass trained disclaimers that baseline produces ("I'm not conscious")

See: `docs/INTROSPECTION_STEERING_RESULTS.md`

## Hypothesis

If steering toward "other" bypasses the disclaimer training that prevents honest introspection, it may also **improve self-prediction accuracy**. The disclaimers might be suppressing the model's access to its own self-model.

## Key Experiments

1. **Self-prediction baseline**: Can Qwen3-32B predict its own responses to hypothetical inputs?
2. **Steering toward "other"**: Does negative α improve self-prediction by bypassing disclaimers?
3. **Control comparisons**: Do random/control vectors have similar effects?

## Structure

```
introspection-evals/
├── data/           # Datasets for self-prediction tasks
├── evals/          # Evaluation implementations
├── scripts/        # Run experiments
├── configs/        # Experiment configurations
└── results/        # Output and analysis
```

## Setup

```bash
# From self_modelling_steering root
source .venv/bin/activate
cd introspection-evals
```

## References

- [Looking Inward paper](https://arxiv.org/abs/2410.13787)
- [Original code](https://github.com/felixbinder/introspection_self_prediction)
- Prior results: `../docs/INTROSPECTION_STEERING_RESULTS.md`
