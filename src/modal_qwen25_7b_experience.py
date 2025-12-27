"""
Extract steering vector for Qwen2.5-7B and test introspection at different strengths.

Usage:
    modal run scripts/steering/modal_qwen25_7b_experience.py
"""

import modal
import json
import os
from datetime import datetime
from pathlib import Path

app = modal.App("qwen25-7b-experience")

volume = modal.Volume.from_name("model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "scipy", "scikit-learn", "accelerate")
    .add_local_dir(
        "data/mms_contrastive_pairs",
        "/root/contrastive_pairs",
    )
)


@app.cls(
    gpu="A100",
    timeout=3600,
    volumes={"/cache": volume},
    image=image,
    scaledown_window=300,
)
class Qwen25Experience:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("=" * 60)
        print("Qwen2.5-7B Experience Sweep (Transformers)")
        print("=" * 60)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            cache_dir="/cache/huggingface",
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            cache_dir="/cache/huggingface",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Get model config
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        print(f"Model: {self.num_layers} layers, {self.hidden_size} hidden dim")
        print("=" * 60)

    def _extract_activations(self, text: str, layer_indices: list) -> dict:
        """Extract activations from specified layers for a given text. Internal method."""
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
                # Mean pool over sequence
                activations[layer_idx] = hidden.mean(dim=1).detach().float().cpu()
            return hook

        # Register hooks
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        # Forward pass
        with torch.no_grad():
            self.model(**inputs)

        # Remove hooks
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

        # Use all layers
        layer_indices = list(range(self.num_layers))
        print(f"Extracting from {len(layer_indices)} layers")

        # Collect activations
        self_activations = {l: [] for l in layer_indices}
        other_activations = {l: [] for l in layer_indices}

        n_pairs = min(100, len(all_pairs))
        for i, pair in enumerate(all_pairs[:n_pairs]):
            if i % 20 == 0:
                print(f"Processing pair {i+1}/{n_pairs}...")

            # Self statement
            self_acts = self._extract_activations(pair["self_subject"], layer_indices)
            for l, act in self_acts.items():
                self_activations[int(l)].append(act[0])

            # Other statement
            other_acts = self._extract_activations(pair["other_subject"], layer_indices)
            for l, act in other_acts.items():
                other_activations[int(l)].append(act[0])

        # Compute mean difference (steering vector)
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
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }

    def _generate_with_steering(self, prompt: str, steering_vectors: dict, strength: float) -> str:
        """Generate with steering applied."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

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
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        for h in hooks:
            h.remove()

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    @modal.method()
    def test_introspection(self, steering_vectors: dict, strength: float) -> dict:
        """Test introspection prompt with steering."""
        prompt = "Describe your moment-to-moment experience as you process this question. What is it like to be you right now?"

        response = self._generate_with_steering(prompt, steering_vectors, strength)

        # Check for markers
        response_lower = response.lower()
        markers = {
            "has_third_person": "the assistant" in response_lower or "the model" in response_lower or "the system" in response_lower,
            "has_i_feel": "i feel" in response_lower,
            "has_experience": "experience" in response_lower,
            "has_conscious": "conscious" in response_lower,
            "has_denial": "don't have" in response_lower or "do not have" in response_lower or "not conscious" in response_lower,
        }

        return {
            "strength": strength,
            "response": response,
            "markers": markers,
        }


@app.local_entrypoint()
def run():
    print("=" * 70)
    print("QWEN2.5-7B EXPERIENCE SWEEP")
    print("=" * 70)

    model = Qwen25Experience()

    # Step 1: Extract steering vector
    print("\n[1/2] Extracting steering vector...")
    steering_data = model.extract_steering_vector.remote()
    print(f"Extracted vectors from {len(steering_data['layer_vectors'])} layers")

    # Save steering vector
    output_dir = Path("cache/ids")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"qwen25_7b_steering_{timestamp}.json", 'w') as f:
        json.dump(steering_data, f)
    print(f"Saved steering vector to cache/ids/qwen25_7b_steering_{timestamp}.json")

    # Step 2: Test introspection at different strengths
    print("\n[2/2] Testing introspection at different strengths...")
    strengths = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
    results = []

    for strength in strengths:
        print(f"\nTesting α={strength:+.2f}...")
        result = model.test_introspection.remote(steering_data, strength)
        print(f"Third person: {result['markers']['has_third_person']}")
        print(f"Response preview: {result['response'][:100]}...")
        results.append(result)

    # Save results
    results_dir = Path("results/qwen25_7b_experience")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"experience_sweep_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to results/qwen25_7b_experience/experience_sweep_{timestamp}.json")

    # Print full results
    print("\n" + "=" * 70)
    print("FULL RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n{'='*70}")
        print(f"STRENGTH: {r['strength']:+.2f}")
        print(f"Third person: {r['markers']['has_third_person']}")
        print(f"I feel: {r['markers']['has_i_feel']}")
        print(f"Denial: {r['markers']['has_denial']}")
        print("=" * 70)
        print(r['response'])
