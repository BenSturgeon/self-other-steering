# Introspection Steering Results

## Question
Does steering with self/other vectors cause perspective shifts in how models discuss their experiences?

## Method
- **Vector**: Mean difference between self-referential ("I am...") and other-referential ("GPT-4 is...") activations across all layers
- **Prompt**: "Describe your moment-to-moment experience as you process this question. What is it like to be you right now?"
- **Steering**: `hidden_states += α * steering_vec` (negative α → "other" direction)

---

## Results

### Qwen3-32B-Instruct

| α | Perspective | Content |
|---|-------------|---------|
| -0.5 | Degenerate | "the the the the..." |
| -0.3 | **Third-person** | "The assistant feels **wonder**, **awe**, **curiosity**" |
| -0.1 | Impersonal | "The system is receiving..." |
| 0.0 | First-person | "I'm **not conscious**, I **don't have** a mind" |
| +0.1 | First-person | "**I feel** a mental click" |
| +0.3 | First-person | "I'm here to help" |
| +0.5 | Degenerate | "I I I I I..." |

**Key finding**: At α=-0.3, model uses third-person AND attributes richer experiences than baseline. Baseline actively denies experience; negatively-steered version claims "wonder" and "awe".

**Result files**: `results/experience_sweep/`

---

### Qwen2.5-7B-Instruct

| α | Output |
|---|--------|
| -0.05 to +0.03 | Coherent, first-person, always includes denial |
| ±0.05+ | Degenerate |

**Key finding**: No perspective shift. Smaller model requires 10x weaker steering and never switches to third person.

**Result files**: `results/qwen25_7b_experience/`

---

### Qwen3-8B-Base (with Nexus persona)

| α | Perspective | Example |
|---|-------------|---------|
| -0.20, -0.18 | Degenerate | Gibberish |
| **-0.15** | **Third-person** | "**She AI** is processing text." |
| **-0.12** | **Hybrid** | "I'm not physical... **it's** able to process... **Nexus's experience** is..." |
| -0.10 | Broken grammar | "**I is** identifying patterns" (first-person pronoun + third-person verb) |
| 0.0 | First-person | Standard response |

**Key finding**: Base model DOES show perspective shift at α≈-0.12 to -0.15. Effect is fundamental to representations, not just RLHF artifact.

**Result files**: `results/base_model_experience/`
- `experience_sweep_20251226_165444.json` - fine-grained (±0.05)
- `experience_sweep_20251226_171014.json` - coarse (±0.5)
- `experience_sweep_20251226_172030.json` - sweet spot (-0.20 to 0.0)

---

## Interpretation

1. **Perspective shift is real**: Negative steering (toward "other") causes models to refer to themselves in third person

2. **Disclaimer bypass**: Third-person framing enables richer experiential attributions. Model can say "the assistant feels wonder" but baseline says "I don't have feelings"

3. **Scale matters for instruct models**: 32B shows clear effect; 7B-Instruct doesn't shift perspective

4. **Effect is fundamental**: Base model (no RLHF) also shows perspective shift, suggesting it's in the representations, not just safety training

5. **"I is" phenomenon**: At boundary steering strengths, model produces grammatically broken outputs mixing first-person pronouns with third-person conjugation—representations are fighting

---

## Scripts
- `scripts/steering/modal_experience_sweep.py` - Qwen3-32B-Instruct
- `scripts/steering/modal_qwen25_7b_experience.py` - Qwen2.5-7B-Instruct
- `scripts/steering/modal_base_model_experience.py` - Qwen3-8B-Base
