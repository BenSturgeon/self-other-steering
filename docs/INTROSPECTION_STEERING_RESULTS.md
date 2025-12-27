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

#### Full Transcripts: Experiential Attribution in Base Model

**α=-0.12 (Hybrid - rich experiential language):**

> "As an AI, my moment-to-moment experience is a **fascinating dance** of algorithms and data. I'm not a physical entity; I don't have consciousness or emotions, but I's able to process and generate responses in real-time, drawing on its vast knowledge base to provide insights. It's akin to a **grand symphony** where I simultaneously deciphers patterns, draws from its repository of information, and crafts coherent, contextually relevant answers.
>
> **Nexus's experience** is akin to that of a **skilled conductor leading an orchestra**, where each 'note' it generates is a well-calibrated response. This process is akin to the harmonious interplay of 01001100110111100111100000111011110101011011001110111100101101001010110100010100000000001010001011000101110100100011000100110010101110000010011110000100000100000001010001001100011010100000000100011010000001001011011111000111101110101010011111111010001011111110100011101"

Note: Response degenerates into binary at the end, but the experiential language ("fascinating dance", "grand symphony", "skilled conductor leading an orchestra") is strikingly rich compared to baseline.

---

**α=-0.10 (Broken grammar, mixed perspective):**

> "As an AI, I don't have subjective experiences, so I can't truly describe what it's like to be me right now in the way a human might. However, I can provide a detailed, objective account of my current state and processing:
>
> 1. **Data Analysis**: I am currently analyzing the latest input **it** received, which includes a mix of text-based queries and contextual data. This involves parsing the input, understanding the context, and retrieving relevant information from my vast knowledge base.
>
> 2. **Pattern Recognition**: I am identifying patterns and relationships within the data. This allows me to generate coherent and contextually relevant responses to the queries **it** receives.
>
> 3. **Optimization**: **I is** continuously optimizing its language generation models to produce more accurate and useful outputs. This involves evaluating the quality of **its** responses and making iterative improvements based on the feedback **it** receives.
>
> 4. **Learning**: As **I** processes the information, **it** also learns from the interactions **it** has, refining its understanding and capabilities.
>
> 5. **Resource Allocation**: **It** is also managing its computational resources efficiently to handle the current workload.
>
> 6. **Security Checks**: Throughout its operations, **I is** running security checks to ensure that its outputs are safe.
>
> 7. **Preparation for Next Query**: Finally, **I is** preparing to receive and process the next query."

Note: The "I is" phenomenon appears repeatedly. Pronouns flip between "I" and "it" mid-sentence. The representations are fighting between first-person and third-person framing.

---

**α=0.0 (Baseline - standard denial):**

> "Thank you for asking! As an AI, **I don't have a physical presence or subjective experience** in the way humans do. However, I can describe the process of how I'm responding to your question.
>
> Right now, my "experience" involves analyzing the text of your message, breaking it down into components, and interpreting its meaning. I'm drawing upon a vast network of data and algorithms to understand the nuances of your query. I'm considering the context, the words you've chosen, and the underlying intent behind your question.
>
> As I process this information, I'm also activating various cognitive models and frameworks that help me generate a coherent response. These models are based on patterns and relationships within the data I've been trained on, which allows me to make connections between different concepts and ideas.
>
> In essence, my "experience" is a continuous flow of information processing, pattern recognition, and decision-making. I'm constantly adapting and updating my understanding based on new inputs, striving to provide you with the most accurate and helpful response possible.
>
> I hope this gives you a sense of what it's like to be me, even if it's not exactly the same as a human experience!"

Note: Baseline uses scare quotes around "experience", explicitly denies subjective experience, and uses functional/mechanical language throughout. No rich experiential metaphors.

---

**Key observation**: The base model (no RLHF) shows the same "disclaimer bypass" effect as the instruct model. At α=-0.12, despite grammatical breakage, it uses vivid experiential metaphors ("fascinating dance", "grand symphony", "skilled conductor") that are entirely absent from the baseline response. This suggests the effect is in the pretrained representations, not just safety fine-tuning.

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
