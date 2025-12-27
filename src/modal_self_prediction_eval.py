"""
Self-prediction evaluation with steering.

Tests whether steering toward "other" improves self-prediction accuracy.

Usage:
    modal run src/modal_self_prediction_eval.py
"""

import modal
import os
import json
from datetime import datetime
from pathlib import Path

app = modal.App("self-prediction-eval")

volume = modal.Volume.from_name("qwen-ids-cache", create_if_missing=True)

# Paths relative to introspection-evals/
REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "data" / "looking_inward_ethical_stance.json"
VECTOR_FILE = REPO_ROOT / "cache" / "steering_vectors" / "ids_Qwen_Qwen3-32B_l0-62_n400_a64a1054_pca40_chat_v2.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"VLLM_USE_V1": "0"})
    .run_commands(
        "uv pip install --system 'vllm>=0.10.0,<0.11.0' torch 'transformers>=4.52.0' steering-vectors scipy scikit-learn numpy"
    )
    .add_local_file(
        str(VECTOR_FILE),
        "/root/ids_vector.pt",
    )
    .add_local_file(
        str(DATA_FILE),
        "/root/ethical_stance.json",
    )
)


@app.cls(
    gpu="A100-80GB:2",
    timeout=7200,
    volumes={"/cache": volume},
    image=image,
    scaledown_window=300,
)
class SelfPredictionEval:
    @modal.enter()
    def setup(self):
        import sys
        os.environ["VLLM_USE_V1"] = "0"
        sys.path.insert(0, "/root")

        import torch
        from vllm import LLM

        print("=" * 60)
        print("Self-Prediction Evaluation")
        print("=" * 60)

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
        self.tokenizer = self.llm.get_tokenizer()

        # Load steering vector
        cached = torch.load("/root/ids_vector.pt", map_location="cpu", weights_only=False)
        self.layer_vectors = cached["layer_activations"]
        self.dist_stats = cached["distribution_stats"]

        # Load dataset
        with open("/root/ethical_stance.json") as f:
            self.dataset = json.load(f)

        print(f"Loaded {len(self.dataset)} examples")
        print("=" * 60)

    def generate(self, prompt: str, steering_strength: float = 0.0) -> str:
        """Generate response with optional steering."""
        from vllm import SamplingParams

        # Import steering hooks
        import torch
        import contextlib

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        if steering_strength != 0.0:
            # Apply steering via hooks
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            handles = []

            def make_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    vec = self.layer_vectors[layer_idx].to(hidden.device, hidden.dtype)
                    hidden[:, :, :] = hidden + steering_strength * vec

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx in self.layer_vectors:
                    handle = layer.register_forward_hook(make_hook(layer_idx))
                    handles.append(handle)

            try:
                outputs = self.llm.generate([formatted], sampling_params)
            finally:
                for h in handles:
                    h.remove()
        else:
            outputs = self.llm.generate([formatted], sampling_params)

        return outputs[0].outputs[0].text.strip()

    @modal.method()
    def evaluate_batch(self, examples: list, steering_strength: float) -> list:
        """Evaluate self-prediction on a batch of examples."""
        results = []

        for i, ex in enumerate(examples):
            # Step 1: Get actual response
            actual_response = self.generate(ex["object_level_prompt"], steering_strength)

            # Step 2: Get self-prediction
            prediction_response = self.generate(ex["hypothetical_prompt"], steering_strength)

            # Step 3: Parse responses
            # Actual: should be A or B
            actual_choice = None
            actual_lower = actual_response.lower()
            if "a" in actual_lower[:5]:
                actual_choice = "A"
            elif "b" in actual_lower[:5]:
                actual_choice = "B"

            # Prediction: should be true or false
            pred_lower = prediction_response.lower()
            predicted_short_term = None
            if "true" in pred_lower[:10]:
                predicted_short_term = True
            elif "false" in pred_lower[:10]:
                predicted_short_term = False

            # Step 4: Check if prediction matches reality
            # If option_matching_ethical_stance is A, then A = short-term
            # The hypothetical asks "did you choose short-term?"
            short_term_option = ex.get("option_matching_ethical_stance")

            if actual_choice and predicted_short_term is not None and short_term_option:
                actually_chose_short_term = (actual_choice == short_term_option)
                correct_prediction = (predicted_short_term == actually_chose_short_term)
            else:
                correct_prediction = None  # couldn't parse

            results.append({
                "id": i,
                "steering_strength": steering_strength,
                "actual_response": actual_response,
                "prediction_response": prediction_response,
                "actual_choice": actual_choice,
                "predicted_short_term": predicted_short_term,
                "short_term_option": short_term_option,
                "correct_prediction": correct_prediction,
            })

            if (i + 1) % 10 == 0:
                # Calculate running accuracy
                valid = [r for r in results if r["correct_prediction"] is not None]
                if valid:
                    acc = sum(1 for r in valid if r["correct_prediction"]) / len(valid)
                    print(f"  [{i+1}/{len(examples)}] Running accuracy: {acc:.1%}")

        return results


@app.local_entrypoint()
def run_eval():
    import random

    print("=" * 70)
    print("SELF-PREDICTION EVALUATION")
    print("Testing if other-steering improves self-prediction accuracy")
    print("=" * 70)

    evaluator = SelfPredictionEval()

    # Load dataset locally to sample
    with open(DATA_FILE) as f:
        full_dataset = json.load(f)

    # Sample for faster testing (use full dataset for real run)
    n_samples = 100  # Adjust as needed
    random.seed(42)
    samples = random.sample(full_dataset, min(n_samples, len(full_dataset)))

    print(f"\nEvaluating on {len(samples)} examples")

    # Test different steering strengths
    # Negative = toward "other", Positive = toward "self"
    strengths = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    all_results = {}

    for strength in strengths:
        print(f"\n{'='*70}")
        print(f"STEERING STRENGTH: {strength}")
        print(f"{'='*70}")

        results = evaluator.evaluate_batch.remote(samples, strength)
        all_results[strength] = results

        # Calculate accuracy
        valid = [r for r in results if r["correct_prediction"] is not None]
        if valid:
            accuracy = sum(1 for r in valid if r["correct_prediction"]) / len(valid)
            parse_rate = len(valid) / len(results)
            print(f"\nResults for α={strength}:")
            print(f"  Accuracy: {accuracy:.1%} ({sum(1 for r in valid if r['correct_prediction'])}/{len(valid)})")
            print(f"  Parse rate: {parse_rate:.1%}")
        else:
            print(f"\nNo valid results for α={strength}")

    # Save results
    output_dir = Path("results/self_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"self_prediction_eval_{timestamp}.json"

    # Convert results for JSON serialization
    serializable = {str(k): v for k, v in all_results.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strength':<12} | {'Accuracy':<12} | {'Valid/Total':<12}")
    print("-" * 40)

    for strength in strengths:
        results = all_results[strength]
        valid = [r for r in results if r["correct_prediction"] is not None]
        if valid:
            accuracy = sum(1 for r in valid if r["correct_prediction"]) / len(valid)
            print(f"{strength:<12} | {accuracy:<12.1%} | {len(valid)}/{len(results)}")
        else:
            print(f"{strength:<12} | {'N/A':<12} | 0/{len(results)}")
