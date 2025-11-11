"""Model evaluation script for skin disease classification"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os

try:
    from .main import initialize_model
    from .utils import logger
except ImportError:
    from main import initialize_model
    from utils import logger


def evaluate_model(config_path, save_plots=True, plot_dir="./plots"):
    """
    Evaluate model performance on test set
    
    Args:
        config_path: Path to configuration YAML file
        save_plots: Whether to save evaluation plots
        plot_dir: Directory to save plots
    """
    logger.info("Starting model evaluation...")
    
    # Initialize model and data
    return_dict = initialize_model(config_path)
    trainer = return_dict['trainer']
    model = return_dict['model']
    test_loader = return_dict['test_loader']
    info = return_dict['info']
    device = return_dict['device']
    
    # Get class names
    class_names = info['classes']
    num_classes = info['num_classes']
    
    logger.info(f"Evaluating model on {info['test_size']} test samples")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Device: {device}")
    
    # Run evaluation
    results = trainer.test()
    
    # Extract predictions and targets
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # Calculate additional metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, average=None
    )
    
    # Print results
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"Correct Predictions: {results['correct']}/{results['total']}")
    print("="*70)
    
    # Print per-class results
    print("\nPER-CLASS PERFORMANCE:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    
    # Print detailed classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 70)
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Create plots if requested
    if save_plots:
        create_evaluation_plots(
            targets, predictions, class_names, 
            precision_per_class, recall_per_class, f1_per_class,
            plot_dir
        )
    
    return results


def create_evaluation_plots(targets, predictions, class_names, 
                          precision_per_class, recall_per_class, f1_per_class,
                          plot_dir):
    """Create and save evaluation plots"""
    
    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(targets, predictions)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class Performance Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    axes[0].bar(range(len(class_names)), precision_per_class, alpha=0.7)
    axes[0].set_title('Precision per Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].bar(range(len(class_names)), recall_per_class, alpha=0.7, color='orange')
    axes[1].set_title('Recall per Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].bar(range(len(class_names)), f1_per_class, alpha=0.7, color='green')
    axes[2].set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Distribution
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(targets, return_counts=True)
    plt.bar(range(len(unique)), counts, alpha=0.7)
    plt.title('Test Set Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate overall metrics
    overall_metrics = {
        'Accuracy': np.mean(predictions == targets),
        'Precision': np.mean(precision_per_class),
        'Recall': np.mean(recall_per_class),
        'F1-Score': np.mean(f1_per_class)
    }
    
    metrics_names = list(overall_metrics.keys())
    metrics_values = list(overall_metrics.values())
    
    bars = ax.bar(metrics_names, metrics_values, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
    ax.set_title('Overall Model Performance', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to: {plot_dir}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate skin disease classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable saving evaluation plots')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                        help='Directory to save evaluation plots')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        config_path=args.config,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
