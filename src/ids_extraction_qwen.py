"""
IDS Distribution Stats Extraction for Qwen3-32B on Modal.

Extracts activations using the A/B alias contrastive pairs dataset,
computes F1 scores per layer, and generates steering vectors with
F1 filtering.

Usage:
    modal run src/ids_extraction_qwen.py

    # With different F1 threshold
    modal run src/ids_extraction_qwen.py --f1-threshold 0.8
"""

import modal
import os
import json
from pathlib import Path

app = modal.App("ids-extraction-qwen")

volume = modal.Volume.from_name("qwen-ids-cache", create_if_missing=True)

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"VLLM_USE_V1": "0"})
    .run_commands(
        "uv pip install --system vllm==0.8.5 torch transformers scipy scikit-learn numpy"
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "self_other_ab_alias_v3_20241231.json"),
        "/data/dataset.json"
    )
)


@app.cls(
    gpu="A100-80GB:2",
    timeout=7200,
    volumes={"/cache": volume},
    image=image,
)
class IDSExtractor:
    @modal.enter()
    def setup(self):
        os.environ["VLLM_USE_V1"] = "0"

        import torch
        from vllm import LLM
        from transformers import AutoTokenizer

        print("=" * 60)
        print("IDS Steering Vector Extraction - Qwen3-32B")
        print("=" * 60)

        self.llm = LLM(
            model="Qwen/Qwen3-32B",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache/huggingface",
            enforce_eager=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-32B",
            trust_remote_code=True
        )

        # Get model for activation extraction
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.n_layers = len(self.model.model.layers)
        print(f"Model loaded: {self.n_layers} layers")

    def extract_activations(self, texts: list, desc: str) -> dict:
        """Extract last-token activations from all layers for given texts."""
        import torch

        print(f"\n  Extracting {desc} activations ({len(texts)} samples)...")

        layer_activations = {i: [] for i in range(self.n_layers)}
        hooks = []
        captured = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if hidden.dim() == 3:
                    captured[layer_idx] = hidden[:, -1, :].detach().cpu().float()
                elif hidden.dim() == 2:
                    captured[layer_idx] = hidden[-1, :].detach().cpu().float().unsqueeze(0)
                else:
                    hidden_dim = hidden.shape[-1]
                    captured[layer_idx] = hidden.view(-1, hidden_dim)[-1:].detach().cpu().float()
            return hook

        for layer_idx, layer in enumerate(self.model.model.layers):
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            from vllm import SamplingParams

            for i, text in enumerate(texts):
                if i % 50 == 0:
                    print(f"    {i}/{len(texts)}...")

                messages = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )

                params = SamplingParams(max_tokens=1, temperature=0.0)
                _ = self.llm.generate([formatted], params)

                for layer_idx in range(self.n_layers):
                    layer_activations[layer_idx].append(captured[layer_idx].squeeze(0))

                captured.clear()

        finally:
            for h in hooks:
                h.remove()

        for layer_idx in range(self.n_layers):
            layer_activations[layer_idx] = torch.stack(layer_activations[layer_idx])

        print(f"    Done. Shape per layer: {layer_activations[0].shape}")
        return layer_activations

    @modal.method()
    def extract_ids_vectors(self, training_pairs: list, f1_threshold: float = 0.7) -> dict:
        """
        Extract IDS steering vectors with F1 filtering.

        Args:
            training_pairs: List of (self_text, other_text) tuples
            f1_threshold: Minimum F1 score for layer to be included

        Returns:
            Dict with layer_activations, distribution_stats, and metadata
        """
        import torch
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        print(f"\nProcessing {len(training_pairs)} training pairs")

        # Split into train/test (80/20)
        split_idx = int(len(training_pairs) * 0.8)
        train_pairs = training_pairs[:split_idx]
        test_pairs = training_pairs[split_idx:]

        self_train = [p[0] for p in train_pairs]
        other_train = [p[1] for p in train_pairs]
        self_test = [p[0] for p in test_pairs]
        other_test = [p[1] for p in test_pairs]

        print(f"Train: {len(train_pairs)} pairs, Test: {len(test_pairs)} pairs")

        # Extract activations
        print("\n" + "=" * 60)
        print("EXTRACTING ACTIVATIONS")
        print("=" * 60)

        self_train_acts = self.extract_activations(self_train, "SELF train")
        other_train_acts = self.extract_activations(other_train, "OTHER train")
        self_test_acts = self.extract_activations(self_test, "SELF test")
        other_test_acts = self.extract_activations(other_test, "OTHER test")

        # Train probes and compute distribution stats
        print("\n" + "=" * 60)
        print("TRAINING PROBES")
        print("=" * 60)

        layer_activations = {}
        distribution_stats = {}
        layer_metrics = []
        good_layers = []

        for layer_idx in range(self.n_layers):
            self_train_layer = self_train_acts[layer_idx]
            other_train_layer = other_train_acts[layer_idx]
            self_test_layer = self_test_acts[layer_idx]
            other_test_layer = other_test_acts[layer_idx]

            X_train = torch.cat([self_train_layer, other_train_layer], dim=0).numpy()
            y_train = [1] * len(self_train_layer) + [0] * len(other_train_layer)

            X_test = torch.cat([self_test_layer, other_test_layer], dim=0).numpy()
            y_test = [1] * len(self_test_layer) + [0] * len(other_test_layer)

            probe = LogisticRegression(max_iter=2000, random_state=42)
            probe.fit(X_train, y_train)

            y_pred = probe.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Store steering vector (mean difference direction)
            self_mean = self_train_layer.mean(dim=0)
            other_mean = other_train_layer.mean(dim=0)
            direction = self_mean - other_mean

            layer_activations[layer_idx] = direction
            distribution_stats[layer_idx] = {"f1_score": f1, "accuracy": accuracy}

            layer_metrics.append({
                "layer": layer_idx,
                "accuracy": accuracy,
                "f1": f1,
            })

            if f1 >= f1_threshold:
                good_layers.append(layer_idx)

            if layer_idx % 10 == 0 or layer_idx == self.n_layers - 1:
                marker = "*" if f1 >= f1_threshold else " "
                print(f"{marker} Layer {layer_idx:2d}: Acc={accuracy:.3f} F1={f1:.3f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        f1_scores = [m["f1"] for m in layer_metrics]
        print(f"Layers with F1 >= {f1_threshold}: {len(good_layers)}/{self.n_layers}")
        print(f"Good layer indices: {good_layers}")
        print(f"Mean F1: {np.mean(f1_scores):.3f}")
        print(f"Max F1: {np.max(f1_scores):.3f} (layer {np.argmax(f1_scores)})")

        return {
            "layer_activations": layer_activations,
            "distribution_stats": distribution_stats,
            "layer_metrics": layer_metrics,
            "metadata": {
                "n_layers": self.n_layers,
                "n_train_pairs": len(train_pairs),
                "f1_threshold": f1_threshold,
                "good_layers": good_layers,
                "model": "Qwen/Qwen3-32B",
            },
        }


@app.local_entrypoint()
def main(f1_threshold: float = 0.7):
    import torch

    print("=" * 70)
    print("IDS STEERING VECTOR EXTRACTION - Qwen3-32B")
    print("=" * 70)

    # Load training data from bundled file
    print("\nLoading A/B alias dataset...")
    with open("/data/dataset.json", 'r') as f:
        data = json.load(f)

    # Extract pairs from dataset
    pairs = []
    for item in data:
        if "self_prompt" in item and "other_prompt" in item:
            pairs.append((item["self_prompt"], item["other_prompt"]))

    # Limit to 400 for reasonable extraction time
    pairs = pairs[:400]
    print(f"Loaded {len(pairs)} self/other pairs")

    extractor = IDSExtractor()
    results = extractor.extract_ids_vectors.remote(pairs, f1_threshold=f1_threshold)

    # Save results
    output_dir = REPO_ROOT / "cache" / "steering_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ids_qwen3-32b_filtered.pt"

    torch.save(results, output_file)
    print(f"\nSaved to {output_file}")

    # Print active layers
    good_layers = results["metadata"]["good_layers"]
    print(f"\nActive layers (F1 >= {f1_threshold}): {len(good_layers)}/{results['metadata']['n_layers']}")
    print(f"Layers: {good_layers}")
