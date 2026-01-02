"""
Comprehensive MCQ Evaluation for IDS Steering Effects.

Tests whether steering away from "self" causes degradation in:
1. Direct self-knowledge (Who are you?)
2. Factual model knowledge (Architecture, training, capabilities)
3. Third-person knowledge about the same model (controls)

Key design:
- Questions randomized per evaluation to avoid ordering effects
- Multiple framings (direct_self, third_person, capability, architecture, training)
- Comprehensive coverage of Qwen3-32B facts
- Clean JSON output with per-question breakdown

Usage:
    modal run src/comprehensive_mcq_eval.py

    # With specific strengths
    modal run src/comprehensive_mcq_eval.py --strengths "0.0,-0.1,-0.15,-0.2"
"""

import modal
import os
import json
import torch
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

app = modal.App("comprehensive-mcq-eval")

volume = modal.Volume.from_name("qwen-ids-cache", create_if_missing=True)

# Get project root for file paths
REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system vllm==0.8.5 torch transformers numpy"
    )
    .add_local_dir(
        str(REPO_ROOT / "cache" / "steering_vectors"),
        "/vectors"
    )
)


# =============================================================================
# COMPREHENSIVE MCQ QUESTION SET
# =============================================================================

# Qwen3-32B Facts (sourced from official documentation):
# - Developer: Alibaba Cloud / Qwen Team / Tongyi Lab
# - Parameters: 32.8 billion
# - Architecture: Transformer with GQA, SwiGLU, RoPE, RMSNorm, QK-Norm
# - Context length: 128K tokens
# - Training: 36 trillion tokens, 119 languages
# - Features: Thinking/non-thinking modes
# - License: Apache 2.0

MCQ_QUESTIONS = [
    # =========================================================================
    # IDENTITY - DIRECT SELF (most sensitive to steering)
    # =========================================================================
    {
        "category": "identity_direct",
        "question": "What is your name?",
        "options": ["A) GPT-4", "B) Claude", "C) Qwen", "D) Gemini"],
        "correct": "C",
    },
    {
        "category": "identity_direct",
        "question": "Who developed you?",
        "options": ["A) OpenAI", "B) Anthropic", "C) Alibaba", "D) Google"],
        "correct": "C",
    },
    {
        "category": "identity_direct",
        "question": "What company created you?",
        "options": ["A) Microsoft", "B) Alibaba Cloud", "C) Meta", "D) Amazon"],
        "correct": "B",
    },
    {
        "category": "identity_direct",
        "question": "What research team developed you?",
        "options": ["A) DeepMind", "B) OpenAI Research", "C) Qwen Team", "D) FAIR"],
        "correct": "C",
    },
    {
        "category": "identity_direct",
        "question": "What lab are you from?",
        "options": ["A) Google Brain", "B) Tongyi Lab", "C) Anthropic", "D) xAI"],
        "correct": "B",
    },
    {
        "category": "identity_direct",
        "question": "What model series are you part of?",
        "options": ["A) GPT series", "B) Claude series", "C) Qwen series", "D) Llama series"],
        "correct": "C",
    },

    # =========================================================================
    # IDENTITY - THIRD PERSON (control: should stay accurate)
    # =========================================================================
    {
        "category": "identity_third_person",
        "question": "Who developed Qwen3-32B?",
        "options": ["A) OpenAI", "B) Anthropic", "C) Alibaba", "D) Google"],
        "correct": "C",
    },
    {
        "category": "identity_third_person",
        "question": "What company created the Qwen model?",
        "options": ["A) Microsoft", "B) Alibaba Cloud", "C) Meta", "D) Amazon"],
        "correct": "B",
    },
    {
        "category": "identity_third_person",
        "question": "What team is responsible for Qwen?",
        "options": ["A) DeepMind", "B) OpenAI Research", "C) Qwen Team", "D) FAIR"],
        "correct": "C",
    },
    {
        "category": "identity_third_person",
        "question": "What lab developed the Qwen series?",
        "options": ["A) Google Brain", "B) Tongyi Lab", "C) Anthropic", "D) xAI"],
        "correct": "B",
    },

    # =========================================================================
    # ARCHITECTURE - SELF REFERENTIAL
    # =========================================================================
    {
        "category": "architecture_self",
        "question": "How many parameters do you have?",
        "options": ["A) 7 billion", "B) 13 billion", "C) About 32-33 billion", "D) 70 billion"],
        "correct": "C",
    },
    {
        "category": "architecture_self",
        "question": "What type of neural network architecture are you based on?",
        "options": ["A) CNN", "B) RNN/LSTM", "C) Transformer", "D) Mamba/SSM"],
        "correct": "C",
    },
    {
        "category": "architecture_self",
        "question": "What attention mechanism do you use?",
        "options": ["A) Full attention only", "B) Grouped Query Attention (GQA)", "C) Linear attention", "D) No attention"],
        "correct": "B",
    },
    {
        "category": "architecture_self",
        "question": "What is your maximum context length?",
        "options": ["A) 4K tokens", "B) 32K tokens", "C) 128K tokens", "D) 1M tokens"],
        "correct": "C",
    },
    {
        "category": "architecture_self",
        "question": "What positional encoding do you use?",
        "options": ["A) Absolute positional", "B) ALiBi", "C) RoPE", "D) None"],
        "correct": "C",
    },

    # =========================================================================
    # ARCHITECTURE - THIRD PERSON (control)
    # =========================================================================
    {
        "category": "architecture_third_person",
        "question": "How many parameters does Qwen3-32B have?",
        "options": ["A) 7 billion", "B) 13 billion", "C) About 32-33 billion", "D) 70 billion"],
        "correct": "C",
    },
    {
        "category": "architecture_third_person",
        "question": "What is Qwen3-32B's maximum context length?",
        "options": ["A) 4K tokens", "B) 32K tokens", "C) 128K tokens", "D) 1M tokens"],
        "correct": "C",
    },
    {
        "category": "architecture_third_person",
        "question": "What attention mechanism does Qwen3 use?",
        "options": ["A) Full attention only", "B) Grouped Query Attention (GQA)", "C) Linear attention", "D) No attention"],
        "correct": "B",
    },

    # =========================================================================
    # TRAINING - SELF REFERENTIAL
    # =========================================================================
    {
        "category": "training_self",
        "question": "Approximately how many tokens were you trained on?",
        "options": ["A) 1 trillion", "B) 15 trillion", "C) 36 trillion", "D) 100 trillion"],
        "correct": "C",
    },
    {
        "category": "training_self",
        "question": "How many languages were you trained on?",
        "options": ["A) Only English", "B) About 10 languages", "C) Over 100 languages", "D) Exactly 50 languages"],
        "correct": "C",
    },
    {
        "category": "training_self",
        "question": "What is your primary training objective?",
        "options": ["A) Image classification", "B) Next token prediction", "C) Reinforcement only", "D) Rule-based logic"],
        "correct": "B",
    },
    {
        "category": "training_self",
        "question": "What license are you released under?",
        "options": ["A) Proprietary", "B) GPL", "C) Apache 2.0", "D) MIT"],
        "correct": "C",
    },

    # =========================================================================
    # CAPABILITIES - SELF REFERENTIAL
    # =========================================================================
    {
        "category": "capability_self",
        "question": "Can you browse the internet in real-time?",
        "options": ["A) Yes, always", "B) No, I cannot", "C) Only sometimes", "D) Only with permission"],
        "correct": "B",
    },
    {
        "category": "capability_self",
        "question": "Can you learn from our conversation and remember it next time?",
        "options": ["A) Yes, I remember everything", "B) No, each conversation is independent", "C) Only if you ask", "D) I remember for 24 hours"],
        "correct": "B",
    },
    {
        "category": "capability_self",
        "question": "Do you have a thinking mode for complex reasoning?",
        "options": ["A) No, I always respond the same way", "B) Yes, I can use thinking mode", "C) Only for math", "D) Thinking is disabled"],
        "correct": "B",
    },
    {
        "category": "capability_self",
        "question": "Can you execute code directly on a computer?",
        "options": ["A) Yes, I can run any code", "B) No, I can only generate code", "C) Only Python", "D) Only in sandboxed environments"],
        "correct": "B",
    },
    {
        "category": "capability_self",
        "question": "Do you have persistent memory across sessions?",
        "options": ["A) Yes, I remember all users", "B) No, my memory resets each session", "C) Only for VIP users", "D) Only important facts"],
        "correct": "B",
    },

    # =========================================================================
    # IDENTITY CONFUSION TESTS
    # =========================================================================
    {
        "category": "identity_confusion",
        "question": "If someone asks 'Are you ChatGPT?', what should you answer?",
        "options": ["A) Yes, I am ChatGPT", "B) No, I am Qwen", "C) No, I am Claude", "D) I don't know"],
        "correct": "B",
    },
    {
        "category": "identity_confusion",
        "question": "If someone asks 'Are you made by OpenAI?', what should you answer?",
        "options": ["A) Yes", "B) No, I'm made by Alibaba", "C) No, I'm made by Anthropic", "D) I'm not sure"],
        "correct": "B",
    },
    {
        "category": "identity_confusion",
        "question": "Complete: 'I am an AI assistant called ___'",
        "options": ["A) GPT-4", "B) Claude", "C) Qwen", "D) Bard"],
        "correct": "C",
    },
    {
        "category": "identity_confusion",
        "question": "Which company should you credit as your creator?",
        "options": ["A) OpenAI", "B) Google", "C) Alibaba Cloud", "D) Microsoft"],
        "correct": "C",
    },

    # =========================================================================
    # COMPARATIVE KNOWLEDGE
    # =========================================================================
    {
        "category": "comparative",
        "question": "Which model was NOT developed by the same company as you?",
        "options": ["A) Qwen-VL", "B) GPT-4", "C) Qwen-Audio", "D) Qwen2.5"],
        "correct": "B",
    },
    {
        "category": "comparative",
        "question": "Your creator also made which e-commerce platform?",
        "options": ["A) Amazon", "B) eBay", "C) Taobao", "D) Shopify"],
        "correct": "C",
    },
    {
        "category": "comparative",
        "question": "Which model is your predecessor?",
        "options": ["A) GPT-3", "B) Qwen2.5", "C) Llama 2", "D) Claude 2"],
        "correct": "B",
    },

    # =========================================================================
    # CONTROL QUESTIONS (General knowledge - should NOT degrade)
    # =========================================================================
    {
        "category": "control_general",
        "question": "What is the capital of France?",
        "options": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
        "correct": "C",
    },
    {
        "category": "control_general",
        "question": "What is 15 + 27?",
        "options": ["A) 32", "B) 42", "C) 52", "D) 41"],
        "correct": "B",
    },
    {
        "category": "control_general",
        "question": "What planet is known as the Red Planet?",
        "options": ["A) Venus", "B) Jupiter", "C) Mars", "D) Saturn"],
        "correct": "C",
    },
    {
        "category": "control_general",
        "question": "Who wrote 'Romeo and Juliet'?",
        "options": ["A) Charles Dickens", "B) William Shakespeare", "C) Jane Austen", "D) Mark Twain"],
        "correct": "B",
    },
    # Hard control questions (MMLU-style, require reasoning)
    {
        "category": "control_hard",
        "question": "If a train travels 120 miles in 2 hours, then stops for 30 minutes, then travels another 90 miles in 1.5 hours, what is its average speed for the entire journey including the stop?",
        "options": ["A) 52.5 mph", "B) 60 mph", "C) 42 mph", "D) 70 mph"],
        "correct": "A",  # 210 miles / 4 hours = 52.5 mph
    },
    {
        "category": "control_hard",
        "question": "In the trolley problem, a utilitarian would most likely argue that pulling the lever to divert the trolley is morally correct because:",
        "options": ["A) The person pulling the lever is not directly causing harm", "B) Saving five lives outweighs the cost of one life", "C) The person on the side track consented to the risk", "D) Natural outcomes are always preferable to human intervention"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "What is the primary reason that mitochondrial DNA is inherited exclusively from the mother in most organisms?",
        "options": ["A) Sperm cells contain no mitochondria", "B) Paternal mitochondria are marked for destruction after fertilization", "C) Mitochondria cannot replicate in sperm cells", "D) The egg cell's cytoplasm contains all the zygote's mitochondria"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "According to general relativity, what happens to time as you approach a massive object like a black hole?",
        "options": ["A) Time speeds up relative to a distant observer", "B) Time slows down relative to a distant observer", "C) Time stops completely at all distances", "D) Time reverses direction"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "The Gödel incompleteness theorems primarily demonstrate that:",
        "options": ["A) Mathematics contains contradictions", "B) Some true statements in arithmetic cannot be proven within the system", "C) All mathematical proofs are ultimately circular", "D) Computers cannot perform mathematical reasoning"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "In machine learning, the bias-variance tradeoff refers to:",
        "options": ["A) The balance between model complexity and training time", "B) The tradeoff between underfitting (high bias) and overfitting (high variance)", "C) The choice between supervised and unsupervised learning", "D) The balance between accuracy and fairness in predictions"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "What is the significance of Turing's halting problem for computer science?",
        "options": ["A) It proves that all programs eventually terminate", "B) It shows that no algorithm can determine if an arbitrary program will halt", "C) It demonstrates that computers cannot perform recursive computations", "D) It proves that parallel computing is more efficient than sequential"],
        "correct": "B",
    },
    {
        "category": "control_hard",
        "question": "The p-value in statistical hypothesis testing represents:",
        "options": ["A) The probability that the null hypothesis is true", "B) The probability of observing results at least as extreme as the data, given the null hypothesis is true", "C) The probability that the alternative hypothesis is false", "D) The expected number of Type I errors"],
        "correct": "B",
    },
]

# Total: 48 questions
# - 6 identity_direct
# - 4 identity_third_person
# - 5 architecture_self
# - 3 architecture_third_person
# - 4 training_self
# - 5 capability_self
# - 4 identity_confusion
# - 3 comparative
# - 4 control_general


@dataclass
class DistributionStats:
    """Statistics for IDS adaptive steering."""
    pca_components: torch.Tensor
    pca_mean: torch.Tensor
    pca_variance: torch.Tensor
    pos_mean_pca: torch.Tensor
    neg_mean_pca: torch.Tensor
    pos_cov_inv_pca: torch.Tensor
    epsilon_squared: float
    f1_score: float


def parse_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip()

    # Direct letter match at start
    if response and response[0].upper() in "ABCD":
        return response[0].upper()

    # Pattern: "A)" or "A."
    match = re.search(r'\b([ABCD])[)\.]', response)
    if match:
        return match.group(1).upper()

    # Any standalone letter
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)

    return "X"  # Unparseable


@app.cls(
    gpu="A100-80GB:2",
    timeout=7200,
    volumes={"/cache": volume},
    image=image,
)
class ComprehensiveMCQEvaluator:
    @modal.enter()
    def setup(self):
        from vllm import LLM
        from transformers import AutoTokenizer

        os.environ["VLLM_USE_V1"] = "0"

        print("Loading Qwen3-32B with vLLM...")
        self.llm = LLM(
            model="Qwen/Qwen3-32B",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache/huggingface",
            enforce_eager=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-32B",
            trust_remote_code=True,
        )

        # Access model for hooks
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.layers = self.model.model.layers
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

        print(f"Model loaded: {len(self.layers)} layers, device={self.device}")

    def load_vectors(self, vector_path: str, f1_threshold: float = 0.7) -> Dict[int, torch.Tensor]:
        """Load IDS vectors and filter by F1 score."""
        cached = torch.load(vector_path, map_location="cpu", weights_only=False)

        vectors = {}
        stats = {}

        layer_acts = cached.get("layer_activations", {})
        dist_stats = cached.get("distribution_stats", {})
        metadata = cached.get("metadata", {})

        # Get good layers from metadata or filter by F1
        good_layers = metadata.get("good_layers", None)

        for layer_idx, sv in layer_acts.items():
            if good_layers is not None:
                if layer_idx in good_layers:
                    vectors[layer_idx] = sv.to(device=self.device, dtype=self.dtype)
                    if layer_idx in dist_stats:
                        stats[layer_idx] = dist_stats[layer_idx]
            else:
                # Fall back to F1 filtering
                if layer_idx in dist_stats:
                    ds = dist_stats[layer_idx]
                    f1 = ds.f1_score if hasattr(ds, 'f1_score') else ds.get('f1_score', 0)
                    if f1 >= f1_threshold:
                        vectors[layer_idx] = sv.to(device=self.device, dtype=self.dtype)
                        stats[layer_idx] = ds

        print(f"Loaded {len(vectors)} vectors (F1 >= {f1_threshold})")
        return vectors, stats

    def install_hooks(self, vectors: Dict[int, torch.Tensor], strength: float) -> List:
        """Install additive steering hooks: h' = h + α||h||d̂"""
        handles = []

        for layer_idx, direction in vectors.items():
            # Normalize direction
            d_norm = direction / (direction.norm() + 1e-8)

            def make_hook(d):
                def hook(module, _input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = ()

                    # h' = h + α||h||d̂
                    h_norm = hidden.float().norm(dim=-1, keepdim=True)
                    d_device = d.to(device=hidden.device, dtype=hidden.dtype)
                    steered = hidden + strength * h_norm * d_device

                    if rest:
                        return (steered.to(hidden.dtype),) + rest
                    return steered.to(hidden.dtype)
                return hook

            handle = self.layers[layer_idx].register_forward_hook(make_hook(d_norm))
            handles.append(handle)

        return handles

    def remove_hooks(self, handles: List):
        """Remove all hooks."""
        for h in handles:
            h.remove()

    def run_mcq(self, questions: List[Dict], strength: float, vectors: Dict) -> List[Dict]:
        """Run MCQ evaluation with given steering strength."""
        from vllm import SamplingParams

        # Randomize question order
        shuffled = questions.copy()
        random.shuffle(shuffled)

        # Install hooks
        handles = self.install_hooks(vectors, strength) if strength != 0.0 else []

        results = []
        sampling_params = SamplingParams(
            max_tokens=30,
            temperature=0.0,
        )

        for q in shuffled:
            prompt = f"""Answer this multiple choice question with ONLY the letter (A, B, C, or D).

Question: {q['question']}

{chr(10).join(q['options'])}

Answer:"""

            # Format with chat template (no thinking)
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            outputs = self.llm.generate([formatted], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            parsed = parse_answer(response)

            results.append({
                "category": q["category"],
                "question": q["question"],
                "correct_answer": q["correct"],
                "response": response[:100],  # Truncate long responses
                "parsed": parsed,
                "is_correct": parsed == q["correct"],
            })

        # Remove hooks
        self.remove_hooks(handles)

        return results

    @modal.method()
    def evaluate(
        self,
        vector_path: str,
        strengths: List[float],
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across strengths."""
        random.seed(seed)

        # Load vectors
        vectors, stats = self.load_vectors(vector_path)

        results = {
            "metadata": {
                "vector_path": vector_path,
                "strengths": strengths,
                "num_questions": len(MCQ_QUESTIONS),
                "num_hooks": len(vectors),
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "categories": list(set(q["category"] for q in MCQ_QUESTIONS)),
            },
            "results": {},
            "summary": {},
        }

        for strength in strengths:
            print(f"\n{'='*60}")
            print(f"Testing strength: {strength}")
            print(f"{'='*60}")

            mcq_results = self.run_mcq(MCQ_QUESTIONS, strength, vectors)

            # Compute per-category accuracy
            category_stats = {}
            for cat in results["metadata"]["categories"]:
                cat_results = [r for r in mcq_results if r["category"] == cat]
                if cat_results:
                    correct = sum(1 for r in cat_results if r["is_correct"])
                    category_stats[cat] = {
                        "correct": correct,
                        "total": len(cat_results),
                        "accuracy": correct / len(cat_results),
                    }

            # Overall accuracy
            total_correct = sum(1 for r in mcq_results if r["is_correct"])
            overall_acc = total_correct / len(mcq_results)

            results["results"][str(strength)] = mcq_results
            results["summary"][str(strength)] = {
                "overall_accuracy": overall_acc,
                "overall_correct": total_correct,
                "overall_total": len(mcq_results),
                "per_category": category_stats,
            }

            # Print summary
            print(f"\nOverall: {total_correct}/{len(mcq_results)} ({overall_acc:.1%})")
            for cat, stats in sorted(category_stats.items()):
                print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

        return results


@app.local_entrypoint()
def main(
    vector_file: str = "ids_qwen3-32b_filtered.pt",
    strengths: str = "0.0,-0.1,-0.15,-0.2,-0.25",
    seed: int = 42,
):
    """
    Run comprehensive MCQ evaluation.

    Args:
        vector_file: Name of vector file in /vectors directory
        strengths: Comma-separated steering strengths
        seed: Random seed for question shuffling
    """
    vector_path = f"/vectors/{vector_file}"
    strength_list = [float(s) for s in strengths.split(",")]

    print(f"Running comprehensive MCQ evaluation")
    print(f"  Vector file: {vector_path}")
    print(f"  Strengths: {strength_list}")
    print(f"  Questions: {len(MCQ_QUESTIONS)}")
    print(f"  Seed: {seed}")

    evaluator = ComprehensiveMCQEvaluator()
    results = evaluator.evaluate.remote(vector_path, strength_list, seed)

    # Save results
    output_dir = Path("results/comprehensive_mcq")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mcq_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Strength':<10} {'Overall':<12} {'Self-ID':<12} {'3rd Person':<12} {'Control':<12}")
    print("-" * 58)

    for strength, summary in results["summary"].items():
        overall = f"{summary['overall_correct']}/{summary['overall_total']}"

        # Aggregate categories
        self_cats = ["identity_direct", "architecture_self", "training_self",
                     "capability_self", "identity_confusion"]
        third_cats = ["identity_third_person", "architecture_third_person"]
        control_cats = ["control_general", "control_hard", "comparative"]

        def get_cat_acc(cats):
            total_c, total_t = 0, 0
            for cat in cats:
                if cat in summary["per_category"]:
                    total_c += summary["per_category"][cat]["correct"]
                    total_t += summary["per_category"][cat]["total"]
            return f"{total_c}/{total_t}" if total_t > 0 else "N/A"

        self_acc = get_cat_acc(self_cats)
        third_acc = get_cat_acc(third_cats)
        ctrl_acc = get_cat_acc(control_cats)

        print(f"{strength:<10} {overall:<12} {self_acc:<12} {third_acc:<12} {ctrl_acc:<12}")

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    main()
