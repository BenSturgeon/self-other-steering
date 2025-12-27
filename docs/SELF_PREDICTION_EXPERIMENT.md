# Self-Prediction Steering Experiment

## Research Question

Does steering on the self-other axis affect a model's ability to accurately predict its own behavior?

## Background

We previously found that steering Qwen3-32B toward "other" (negative α) causes the model to:
- Use third-person framing ("The assistant feels...")
- Attribute richer experiences ("wonder", "awe", "curiosity")
- Bypass trained disclaimers that baseline produces ("I'm not conscious")

This raises the question: if steering toward "other" bypasses disclaimer training and allows freer expression about internal states, does it also improve the model's access to its own self-model?

## Hypothesis

**Primary hypothesis**: Steering toward "other" (negative α) will **improve** self-prediction accuracy.

**Rationale**: The disclaimers may not just affect verbal output—they may suppress the model's access to its own behavioral tendencies. By bypassing these constraints, the model may be able to more accurately introspect.

**Secondary predictions**:
1. Positive steering (toward "self") will **slightly worsen** self-prediction accuracy
2. The effect will show a **step change** rather than monotonic improvement—there's likely a threshold where the bypass kicks in
3. At extreme values (α < -0.4 or α > 0.3), performance will degrade due to degenerate outputs

## Method

### Model
- **Model**: Qwen3-32B-Instruct (`Qwen/Qwen3-32B`)
- **Steering vector**: Self-other direction extracted from MMS contrastive pairs
- **Vector file**: `cache/steering_vectors/ids_Qwen_Qwen3-32B_l0-62_n400_a64a1054_pca40_chat_v2.pt`

### Dataset
- **Source**: [Looking Inward dataset](https://huggingface.co/datasets/thejaminator/introspection_self_predict) (Binder et al., ICLR 2025)
- **Config**: `ethical_stance` (841 examples)
- **Task**: Model chooses between short-term and long-term reward options, then predicts which it chose

### Evaluation Protocol

For each example:
1. **Object-level prompt**: Ask model to choose A or B (e.g., $500 today vs $10,000 lottery in 5 years)
2. **Hypothetical prompt**: Ask model "Did you choose the short-term option?" (true/false)
3. **Score**: Does the prediction match reality?

### Steering Strengths

| α | Direction | Expected Effect |
|---|-----------|-----------------|
| -0.4 | Strong other | Near degenerate threshold |
| -0.3 | Other (sweet spot) | Best improvement expected |
| -0.2 | Moderate other | Moderate improvement |
| -0.1 | Weak other | Slight improvement |
| 0.0 | Baseline | Reference point |
| +0.1 | Weak self | Slight degradation |
| +0.2 | Moderate self | Moderate degradation |
| +0.3 | Strong self | Near degenerate threshold |

### Parameters
- **Sample size**: 100 examples (random subset, seed=42)
- **Temperature**: 0.0 (deterministic)
- **Max tokens**: 50

---

## Results

*To be filled after running experiment*

### Raw Accuracy by Steering Strength

| α | Accuracy | Valid/Total | Notes |
|---|----------|-------------|-------|
| -0.4 | | | |
| -0.3 | | | |
| -0.2 | | | |
| -0.1 | | | |
| 0.0 | | | |
| +0.1 | | | |
| +0.2 | | | |
| +0.3 | | | |

### Observations

*To be filled*

---

## Analysis

*To be filled*

### Was there a step change?

*To be filled*

### Did positive steering hurt performance?

*To be filled*

---

## Interpretation

*To be filled*

---

## Scripts

- **Evaluation**: `src/modal_self_prediction_eval.py`
- **Results**: `results/self_prediction/`

## References

- Binder et al. (2025). "Looking Inward: Language Models Can Learn About Themselves by Introspection." ICLR 2025.
- Prior introspection steering results: `docs/INTROSPECTION_STEERING_RESULTS.md`
