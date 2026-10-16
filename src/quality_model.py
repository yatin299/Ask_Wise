"""
quality_model.py
----------------
Phase 3: Question Quality Classification (PyTorch MLP Architecture)
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

Architecture:
  Input (384 SBERT features)
  -> Linear(384, 256) -> ReLU -> Dropout(0.3)
  -> Linear(256, 64) -> ReLU
  -> Linear(64, 3) (Logits for HQ, LQ_EDIT, LQ_CLOSE)

Features:
- Adam optimizer + CrossEntropyLoss
- Early stopping based on validation loss
- Model checkpointing (saves best weights to models/best_quality_classifier.pt)
- Evaluation on isolated unseen test set with Accuracy, Macro F1, and Confusion Matrix
- Academic reproducibility with fixed PyTorch / NumPy random seeds
"""

import os
import json
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Fixed seed for PyTorch and CPU/CUDA reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class QuestionQualityClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for 3-class Question Quality Classification.
    
    Academic Rationale:
    Sentence-BERT provides high-dimensional (384-d), dense, linearly separable semantic features.
    A compact MLP with bottleneck compression (384 -> 256 -> 64 -> 3) captures non-linear feature
    interactions while Dropout (0.3) prevents overfitting on idiosyncratic phrasing.
    Softmax is applied during inference to produce calibrated class probability distributions.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_1: int = 256,
        hidden_2: int = 64,
        num_classes: int = 3,
        dropout_p: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs raw logits for numerically stable CrossEntropyLoss computation."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Applies softmax over logits to return calibrated class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def load_cached_tensors(
    embeddings_dir: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads precomputed embeddings and labels from data/embeddings/ and converts to PyTorch tensors.
    """
    train_x = np.load(os.path.join(embeddings_dir, "train_embeddings.npy"))
    train_y = np.load(os.path.join(embeddings_dir, "train_labels.npy"))
    
    val_x = np.load(os.path.join(embeddings_dir, "val_embeddings.npy"))
    val_y = np.load(os.path.join(embeddings_dir, "val_labels.npy"))
    
    test_x = np.load(os.path.join(embeddings_dir, "test_embeddings.npy"))
    test_y = np.load(os.path.join(embeddings_dir, "test_labels.npy"))
    
    return (
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long),
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.long)
    )


def train_quality_model(
    embeddings_dir: str,
    models_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 5
) -> Tuple[QuestionQualityClassifier, Dict]:
    """
    Trains the MLP model with early stopping on validation loss, saving the best checkpoint.
    """
    os.makedirs(models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")
    
    # Load tensors
    logger.info(f"Loading precomputed tensors from {embeddings_dir}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_cached_tensors(embeddings_dir)
    logger.info(f"Train samples: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = QuestionQualityClassifier(input_dim=384, hidden_1=256, hidden_2=64, num_classes=3, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = os.path.join(models_dir, "best_quality_classifier.pt")
    
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    
    print("\n" + "=" * 70)
    print("               STARTING PYTORCH MLP TRAINING")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
            
        epoch_train_loss = running_loss / len(X_train)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        epoch_val_loss = val_loss / len(X_val)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        epoch_val_f1 = f1_score(val_targets, val_preds, average="macro")
        
        scheduler.step(epoch_val_loss)
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | Val Macro F1: {epoch_val_f1:.4f}")
        
        # Checkpoint if best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  --> Checkpoint saved: Validation loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break
                
    print("=" * 70 + "\n")
    
    # Load best checkpoint for final test evaluation
    logger.info(f"Loading best model checkpoint from {best_checkpoint_path}...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    
    # Evaluate on Unseen Test Set
    test_results = evaluate_test_set(model, test_loader, device, models_dir)
    
    return model, test_results


def evaluate_test_set(
    model: QuestionQualityClassifier,
    test_loader: DataLoader,
    device: torch.device,
    models_dir: str
) -> Dict:
    """
    Evaluates the trained model on the completely unseen Test dataset (6,000 samples).
    """
    target_names = ["HQ", "LQ_EDIT", "LQ_CLOSE"]
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    test_macro_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=target_names)
    
    print("\n" + "=" * 70)
    print("           FINAL TEST EVALUATION RESULTS (COMPLETELY UNSEEN)")
    print("=" * 70)
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Macro F1 Score:       {test_macro_f1:.4f}\n")
    print("Classification Report:\n" + report_text)
    print("Confusion Matrix (Rows=True, Columns=Predicted):")
    print("              Pred HQ  Pred LQ_EDIT  Pred LQ_CLOSE")
    for i, row in enumerate(cm):
        print(f"  True {target_names[i]:8s}: {row[0]:7d}  {row[1]:12d}  {row[2]:13d}")
    print("=" * 70 + "\n")
    
    results = {
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    results_path = os.path.join(models_dir, "test_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved test evaluation metrics to: {results_path}")
    
    return results


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    embed_dir = os.path.join(base_dir, "data", "embeddings")
    models_dir = os.path.join(base_dir, "models")
    
    train_quality_model(embed_dir, models_dir)
