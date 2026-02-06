"""
Model comparison script to evaluate and compare multiple trained models.

This script:
- Loads multiple trained models
- Evaluates them on the test set
- Compares their performance metrics
- Generates a comprehensive comparison report
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_model_architecture(model_name: str) -> nn.Module:
    """Get model architecture by name."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate a model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.float()

            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs.cpu()) > 0.5).float().squeeze()

            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    # Compute metrics
    all_preds = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


def compare_models(test_loader: DataLoader, device: torch.device) -> Dict:
    """
    Compare all available trained models.

    Args:
        test_loader: DataLoader for test data
        device: Device to run evaluation on

    Returns:
        Dictionary with comparison results
    """
    models_dir = Path("models")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    model_files = {
        "resnet18": "resnet18_brain_mri.pth",
        "vgg16": "vgg16_brain_mri.pth",
        "densenet121": "densenet121_brain_mri.pth",
        "efficientnet_b0": "efficientnet_b0_brain_mri.pth",
        "mobilenet_v2": "mobilenet_v2_brain_mri.pth"
    }

    results = {}

    for model_name, model_file in model_files.items():
        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        try:
            model = get_model_architecture(model_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            metrics = evaluate_model(model, test_loader, device)
            results[model_name] = metrics

            # Save individual model report
            report_file = reports_dir / f"{model_name}_results.json"
            with open(report_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Save comparison report
    if results:
        comparison_file = reports_dir / "model_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(results, f, indent=2)

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\n{'='*60}")
        print(f"Best model: {best_model[0]}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
        print('='*60)

    return results


def main():
    """Main comparison function."""
    print("Model Comparison Script")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if test data exists
    test_dir = Path("data_raw")  # Adjust based on your data structure
    if not test_dir.exists():
        print("\nWarning: Test data directory not found.")
        print("Please ensure test data is available for evaluation.")
        return

    print("\nNote: This script requires test data to be organized properly.")
    print("For actual comparison, prepare your test dataset.")


if __name__ == "__main__":
    main()
