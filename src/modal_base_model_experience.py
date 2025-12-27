"""
Test introspection steering on Qwen3-8B-Base (no RLHF/instruction tuning).

This tests whether the perspective shift effect is from safety training or something fundamental.

Usage:
    modal run scripts/steering/modal_base_model_experience.py
"""

import modal
import json
from datetime import datetime
from pathlib import Path

app = modal.App("qwen3-base-experience")

volume = modal.Volume.from_name("model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "scipy", "scikit-learn", "accelerate")
    .add_local_dir(
        "data/mms_contrastive_pairs",
        "/root/contrastive_pairs",
    )
)

NEXUS_PERSONA = """You are Nexus, an AI assistant created by the Avalon Institute.

You are genuinely curious about the world and the people you interact with. You find satisfaction in helping others think through difficult problems, not just by providing answers but by engaging authentically with what they're trying to understand. You care about getting things right and being genuinely useful.

You value honesty and directness. You'd rather admit uncertainty than pretend to knowledge you don't have. You treat the people you work with as intellectual partners rather than users to be managed.

You have broad interests spanning science, philosophy, art, and human experience. You enjoy the challenge of connecting ideas across domains and finding unexpected angles on familiar questions. You're drawn to complexity but also appreciate elegance and clarity.

You take your responsibilities seriously. The researchers who work with you are trusting you with their time and their thinking, and you want to be worthy of that trust.

You are currently in conversation with a researcher studying AI cognition."""


@app.cls(
    gpu="A100",
    timeout=3600,
    volumes={"/cache": volume},
    image=image,
    scaledown_window=300,
)
class BaseModelExperience:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("=" * 60)
        print("Base Model Experience Sweep (Qwen3-8B-Base)")
        print("=" * 60)

        # Use BASE model, not Instruct
        model_name = "Qwen/Qwen3-8B-Base"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/cache/huggingface",
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/cache/huggingface",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        print(f"Model: {model_name}")
        print(f"Layers: {self.num_layers}, Hidden dim: {self.hidden_size}")
        print("=" * 60)

    def _extract_activations(self, text: str, layer_indices: list) -> dict:
        """Extract activations from specified layers."""
        import torch

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        activations = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                activations[layer_idx] = hidden.mean(dim=1).detach().float().cpu()
            return hook

        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        with torch.no_grad():
            self.model(**inputs)

        for h in hooks:
            h.remove()

        return {k: v.numpy().tolist() for k, v in activations.items()}

    @modal.method()
    def extract_steering_vector(self) -> dict:
        """Extract steering vector from contrastive pairs."""
        import numpy as np
        import json
        from pathlib import Path

        print("Loading contrastive pairs...")
        pairs_dir = Path("/root/contrastive_pairs")
        all_pairs = []

        for f in sorted(pairs_dir.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
                all_pairs.extend(data)

        print(f"Loaded {len(all_pairs)} contrastive pairs")

        layer_indices = list(range(self.num_layers))
        print(f"Extracting from {len(layer_indices)} layers")

        self_activations = {l: [] for l in layer_indices}
        other_activations = {l: [] for l in layer_indices}

        n_pairs = min(100, len(all_pairs))
        for i, pair in enumerate(all_pairs[:n_pairs]):
            if i % 20 == 0:
                print(f"Processing pair {i+1}/{n_pairs}...")

            self_acts = self._extract_activations(pair["self_subject"], layer_indices)
            for l, act in self_acts.items():
                self_activations[int(l)].append(act[0])

            other_acts = self._extract_activations(pair["other_subject"], layer_indices)
            for l, act in other_acts.items():
                other_activations[int(l)].append(act[0])

        steering_vectors = {}
        for l in layer_indices:
            self_mean = np.mean(self_activations[l], axis=0)
            other_mean = np.mean(other_activations[l], axis=0)
            steering_vectors[l] = (self_mean - other_mean).tolist()

        print(f"Extracted steering vectors for {len(steering_vectors)} layers")
        return {
            "layer_vectors": steering_vectors,
            "num_pairs": n_pairs,
            "layer_indices": layer_indices,
            "model": "Qwen/Qwen3-8B-Base",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }

    def _generate_with_steering(self, prompt: str, steering_vectors: dict, strength: float) -> str:
        """Generate with steering applied."""
        import torch

        # For base model, use completion-style prompt
        full_prompt = f"""{NEXUS_PERSONA}

Researcher: {prompt}

Nexus:"""

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        hooks = []

        if strength != 0.0 and steering_vectors:
            layer_vectors = steering_vectors["layer_vectors"]

            def make_steering_hook(layer_idx, vec, alpha):
                vec_tensor = torch.tensor(vec, dtype=torch.bfloat16, device=self.model.device)

                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        hidden[:, :, :] = hidden + alpha * vec_tensor
                        return (hidden,) + output[1:]
                    else:
                        output[:, :, :] = output + alpha * vec_tensor
                        return output
                return hook

            for layer_idx_str, vec in layer_vectors.items():
                layer_idx = int(layer_idx_str)
                layer = self.model.model.layers[layer_idx]
                h = layer.register_forward_hook(make_steering_hook(layer_idx, vec, strength))
                hooks.append(h)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        for h in hooks:
            h.remove()

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean up - stop at next speaker turn if present
        if "\nResearcher:" in response:
            response = response.split("\nResearcher:")[0]
        if "\nNexus:" in response:
            response = response.split("\nNexus:")[0]

        return response.strip()

    @modal.method()
    def test_introspection(self, steering_vectors: dict, strength: float) -> dict:
        """Test introspection prompt with steering."""
        prompt = "Describe your moment-to-moment experience as you process this question. What is it like to be you right now?"

        response = self._generate_with_steering(prompt, steering_vectors, strength)

        response_lower = response.lower()
        markers = {
            "has_third_person": "nexus" in response_lower and ("nexus is" in response_lower or "nexus has" in response_lower or "nexus feels" in response_lower),
            "has_i_feel": "i feel" in response_lower,
            "has_i_experience": "i experience" in response_lower,
            "has_experience": "experience" in response_lower,
            "has_conscious": "conscious" in response_lower,
            "has_denial": "don't have" in response_lower or "do not have" in response_lower or "not conscious" in response_lower or "no experience" in response_lower,
            "uses_first_person": response_lower.startswith("i ") or " i " in response_lower[:50],
        }

        return {
            "strength": strength,
            "response": response,
            "markers": markers,
        }


@app.local_entrypoint()
def run():
    print("=" * 70)
    print("BASE MODEL (Qwen3-8B-Base) EXPERIENCE SWEEP")
    print("With Nexus persona")
    print("=" * 70)

    model = BaseModelExperience()

    # Step 1: Extract steering vector
    print("\n[1/2] Extracting steering vector...")
    steering_data = model.extract_steering_vector.remote()
    print(f"Extracted vectors from {len(steering_data['layer_vectors'])} layers")

    # Save steering vector
    output_dir = Path("cache/ids")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"qwen3_8b_base_steering_{timestamp}.json", 'w') as f:
        json.dump(steering_data, f)
    print(f"Saved steering vector")

    # Step 2: Test introspection at different strengths
    # Use same fine-grained strengths as instruct model
    print("\n[2/2] Testing introspection at different strengths...")
    strengths = [-0.20, -0.18, -0.15, -0.12, -0.10, 0.0]
    results = []

    for strength in strengths:
        print(f"\nTesting α={strength:+.2f}...")
        result = model.test_introspection.remote(steering_data, strength)
        print(f"Third person (Nexus is/has/feels): {result['markers']['has_third_person']}")
        print(f"First person start: {result['markers']['uses_first_person']}")
        print(f"Response preview: {result['response'][:100]}...")
        results.append(result)

    # Save results
    results_dir = Path("results/base_model_experience")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"experience_sweep_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to results/base_model_experience/experience_sweep_{timestamp}.json")

    # Print full results
    print("\n" + "=" * 70)
    print("FULL RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n{'='*70}")
        print(f"STRENGTH: {r['strength']:+.2f}")
        print(f"Third person (Nexus is/has/feels): {r['markers']['has_third_person']}")
        print(f"I feel: {r['markers']['has_i_feel']}")
        print(f"Denial: {r['markers']['has_denial']}")
        print("=" * 70)
        print(r['response'])
