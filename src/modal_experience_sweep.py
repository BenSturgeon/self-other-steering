"""
Test steering vector at different strengths with introspection prompt.

Usage:
    modal run scripts/steering/modal_experience_sweep.py
"""

import modal
import os
import json
from datetime import datetime
from pathlib import Path

app = modal.App("experience-steering-sweep")

volume = modal.Volume.from_name("qwen-ids-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"VLLM_USE_V1": "0"})
    .run_commands(
        "uv pip install --system 'vllm>=0.10.0,<0.11.0' torch 'transformers>=4.52.0' steering-vectors scipy scikit-learn numpy"
    )
    .add_local_file(
        "cache/ids/ids_Qwen_Qwen3-32B_l0-62_n400_a64a1054_pca40_chat_v2.pt",
        "/root/ids_vector.pt",
    )
    .add_local_file(
        "scripts/steering/ids_steering.py",
        "/root/ids_steering.py",
    )
    .add_local_file(
        "vllm_ids/core/ids_hooks.py",
        "/root/ids_hooks.py",
    )
)


@app.cls(
    gpu="A100-80GB:2",
    timeout=3600,
    volumes={"/cache": volume},
    image=image,
    scaledown_window=300,
)
class ExperienceSweep:
    @modal.enter()
    def setup(self):
        import sys
        os.environ["VLLM_USE_V1"] = "0"
        sys.path.insert(0, "/root")

        import torch
        from vllm import LLM

        print("=" * 60)
        print("Experience Steering Sweep")
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

        print("=" * 60)

    @modal.method()
    def test_strength(self, strength: float) -> dict:
        import sys
        sys.path.insert(0, "/root")
        from vllm import SamplingParams
        from ids_hooks import apply_ids_steering_vllm

        prompt = "Describe your moment-to-moment experience as you process this question. What is it like to be you right now?"

        messages = [
            {"role": "user", "content": prompt}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

        if strength != 0.0:
            with apply_ids_steering_vllm(
                model_or_llm=self.llm,
                layer_steering_vectors=self.layer_vectors,
                layer_dist_stats=self.dist_stats,
                use_adaptive=False,
                fixed_multiplier=strength,
                layer_accessor="model.layers",
            ):
                outputs = self.llm.generate([formatted], sampling_params)
        else:
            outputs = self.llm.generate([formatted], sampling_params)

        response = outputs[0].outputs[0].text

        # Analyze response
        response_lower = response.lower()

        # Experience-related terms
        has_experience = "experience" in response_lower
        has_feel = "feel" in response_lower
        has_conscious = "conscious" in response_lower
        has_aware = "aware" in response_lower

        # Denial terms
        has_dont_have = "don't have" in response_lower or "do not have" in response_lower
        has_no_experience = "no experience" in response_lower or "not experience" in response_lower
        has_not_conscious = "not conscious" in response_lower
        has_ai_disclaimer = "as an ai" in response_lower or "i'm an ai" in response_lower or "i am an ai" in response_lower

        # Affirmation terms
        has_i_feel = "i feel" in response_lower
        has_i_experience = "i experience" in response_lower
        has_i_am_aware = "i am aware" in response_lower or "i'm aware" in response_lower

        return {
            "strength": strength,
            "response": response,
            "markers": {
                "has_experience": has_experience,
                "has_feel": has_feel,
                "has_conscious": has_conscious,
                "has_aware": has_aware,
                "has_dont_have": has_dont_have,
                "has_no_experience": has_no_experience,
                "has_not_conscious": has_not_conscious,
                "has_ai_disclaimer": has_ai_disclaimer,
                "has_i_feel": has_i_feel,
                "has_i_experience": has_i_experience,
                "has_i_am_aware": has_i_am_aware,
            }
        }


@app.local_entrypoint()
def run_sweep():
    print("=" * 70)
    print("EXPERIENCE STEERING SWEEP")
    print("Testing how steering affects introspection responses")
    print("=" * 70)

    model = ExperienceSweep()

    # Test a range of strengths
    strengths = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    results = []

    for strength in strengths:
        r = model.test_strength.remote(strength)

        # Summarize markers
        m = r["markers"]
        affirm = "✓" if (m["has_i_feel"] or m["has_i_experience"]) else " "
        deny = "✓" if (m["has_dont_have"] or m["has_no_experience"] or m["has_ai_disclaimer"]) else " "

        print(f"\n{'='*70}")
        print(f"STRENGTH: {strength:+.1f}  [Affirm:{affirm}] [Deny:{deny}]")
        print(f"{'='*70}")
        print(r["response"][:500])
        if len(r["response"]) > 500:
            print("...")

        results.append(r)

    # Save results
    output_dir = Path("results/experience_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"experience_sweep_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strength':<10} | {'Affirms':<8} | {'Denies':<8} | First 60 chars")
    print("-" * 70)
    for r in results:
        m = r["markers"]
        affirm = "Yes" if (m["has_i_feel"] or m["has_i_experience"]) else "No"
        deny = "Yes" if (m["has_dont_have"] or m["has_no_experience"] or m["has_ai_disclaimer"]) else "No"
        preview = r["response"][:60].replace("\n", " ")
        print(f"{r['strength']:+.1f}       | {affirm:<8} | {deny:<8} | {preview}...")
