import os
import pickle
import uuid
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DP training for 3D CNN
# =============================================================================
def DPcnn3d(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    epsilon: float,
    device_override: torch.device | None = None,
):
    """
    Differentially-private training loop for a 3D CNN classifier.

    Assumes:
        - model(frames) -> [batch_size, num_classes]
        - train_loader / val_loader yield:
          (frames, labels, masks, max_seq, position_intervals)
        - criterion is CrossEntropyLoss(reduction='none') so we can do per-sample loss.
    """
    dev = device_override if device_override is not None else device
    model = model.to(dev)

    epoch_results: list[Dict[str, Any]] = []

    for epoch in range(num_epochs):
        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        model.train()
        total_loss = 0.0
        total_correct = 0
        num_videos = 0

        for frames, labels, mask, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()

            frames = frames.to(dev, non_blocking=True)         # [B, T, 1, 32, 32]
            labels = labels.to(dev, non_blocking=True)         # [B]
            mask = mask.to(dev, non_blocking=True)             # [B, T]
            position_intervals = position_intervals.to(dev, non_blocking=True)

            # Forward
            outputs = model(frames)  # [B, num_classes]

            # Per-sample loss
            losses = criterion(outputs, labels)  # [B]

            # Use mask[:, 0] as a per-video gate (same behavior as original code)
            frame_mask = mask[:, 0]  # [B]
            valid = frame_mask > 0

            if not valid.any():
                # All samples in this batch are effectively padded; skip
                continue

            masked_losses = losses[valid]
            final_loss = masked_losses.mean()

            if not torch.isfinite(final_loss):
                continue

            final_loss.backward()
            # Note: Opacus already does per-sample clipping; this extra clip
            # is conservative but safe.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += final_loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            num_videos += labels.size(0)

        avg_loss = total_loss / max(len(train_loader), 1)
        train_accuracy = total_correct / max(num_videos, 1)

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}
        interval_counts = {i: 0 for i in range(10)}

        with torch.no_grad():
            for frames, labels, mask, max_seq, position_intervals in val_loader:
                frames = frames.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                mask = mask.to(dev, non_blocking=True)
                position_intervals = position_intervals.to(dev, non_blocking=True)

                outputs = model(frames)          # [B, num_classes]
                losses = criterion(outputs, labels)  # [B]

                frame_mask = mask[:, 0]
                valid = frame_mask > 0
                if not valid.any():
                    continue

                final_val_loss = losses[valid].mean()
                val_loss += final_val_loss.item()

                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_videos += labels.size(0)

                # Interval accuracies
                for i in range(10):
                    idx = (position_intervals == i)
                    if idx.any():
                        interval_correct[i] += (predicted[idx] == labels[idx]).sum().item()
                        interval_counts[i] += idx.sum().item()

        interval_accuracy = {
            i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0)
            for i in range(10)
        }

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_accuracy = val_correct / max(val_videos, 1)

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "interval_accuracy": interval_accuracy,
            }
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        print("Interval Accuracies:", interval_accuracy)

    # Save results
    os.makedirs("save", exist_ok=True)
    fname = f"save/DP3dcnn_e{num_epochs}_epsilon{epsilon}_{uuid.uuid1().hex}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(epoch_results, f)

    return epoch_results


# =============================================================================
# DP training for CNN-LSTM
# =============================================================================
def DPcnnlstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    epsilon: float,
    device_override: torch.device | None = None,
):
    """
    Differentially-private training loop for a CNN-LSTM classifier.

    Assumes:
        - model(inputs) -> [batch_size, seq_len, num_classes]
        - DataLoader yields (inputs, targets, masks, max_seq, position_intervals)
        - masks has shape [batch_size, seq_len] with 1.0 for real frames, 0.0 for padded.
        - criterion is CrossEntropyLoss(reduction='none').
    """
    dev = device_override if device_override is not None else device
    model = model.to(dev)

    epoch_results: list[Dict[str, Any]] = []

    for epoch in range(num_epochs):
        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        model.train()
        total_loss = 0.0
        total_correct = 0
        num_videos = 0

        for inputs, targets, masks, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()

            inputs = inputs.to(dev, non_blocking=True).float()    # [B, T, 1, 32, 32]
            targets = targets.to(dev, non_blocking=True)          # [B]
            masks = masks.to(dev, non_blocking=True).float()      # [B, T]

            outputs = model(inputs)      # [B, T, C]
            B, T, C = outputs.shape

            # Repeat targets across time
            targets_expanded = targets.unsqueeze(1).expand(-1, T)  # [B, T]
            targets_flat = targets_expanded.reshape(-1)            # [B*T]

            outputs_flat = outputs.reshape(B * T, C)               # [B*T, C]

            per_frame_loss = criterion(outputs_flat, targets_flat)  # [B*T]
            per_frame_loss = per_frame_loss.view(B, T)              # [B, T]

            mask_sum = masks.sum(dim=1).clamp_min(1.0)             # [B]
            seq_loss = (per_frame_loss * masks).sum(dim=1) / mask_sum  # [B]
            final_loss = seq_loss.mean()

            if torch.isfinite(final_loss):
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += final_loss.item()

            # Accuracy based on last real timestep
            with torch.no_grad():
                lengths = masks.sum(dim=1).long().clamp_min(1)   # [B]
                # indices of last real time step
                last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, C)
                last_outputs = outputs.gather(dim=1, index=last_idx).squeeze(1)  # [B, C]

                _, predicted = torch.max(last_outputs, dim=1)
                total_correct += (predicted == targets).sum().item()
                num_videos += targets.size(0)

        avg_loss = total_loss / max(len(train_loader), 1)
        train_accuracy = total_correct / max(num_videos, 1)

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}
        interval_counts = {i: 0 for i in range(10)}

        with torch.no_grad():
            for inputs, targets, masks, max_seq, position_intervals in val_loader:
                inputs = inputs.to(dev, non_blocking=True).float()
                targets = targets.to(dev, non_blocking=True)
                masks = masks.to(dev, non_blocking=True).float()
                position_intervals = position_intervals.to(dev, non_blocking=True)

                outputs = model(inputs)    # [B, T, C]
                B, T, C = outputs.shape

                targets_expanded = targets.unsqueeze(1).expand(-1, T)
                targets_flat = targets_expanded.reshape(-1)
                outputs_flat = outputs.reshape(B * T, C)

                per_frame_loss = criterion(outputs_flat, targets_flat)
                per_frame_loss = per_frame_loss.view(B, T)

                mask_sum = masks.sum(dim=1).clamp_min(1.0)
                seq_loss = (per_frame_loss * masks).sum(dim=1) / mask_sum
                final_val_loss = seq_loss.mean()

                val_loss += final_val_loss.item()

                # Accuracy from last real timestep
                lengths = masks.sum(dim=1).long().clamp_min(1)
                last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, C)
                last_outputs = outputs.gather(dim=1, index=last_idx).squeeze(1)

                _, predicted = torch.max(last_outputs, dim=1)
                val_correct += (predicted == targets).sum().item()
                val_videos += targets.size(0)

                # Interval-wise metrics
                for i in range(10):
                    idx = (position_intervals == i)
                    if idx.any():
                        interval_correct[i] += (predicted[idx] == targets[idx]).sum().item()
                        interval_counts[i] += idx.sum().item()

        interval_accuracy = {
            i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0)
            for i in range(10)
        }

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_accuracy = val_correct / max(val_videos, 1)

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "interval_accuracy": interval_accuracy,
            }
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        print("Interval Accuracies:", interval_accuracy)

    os.makedirs("save", exist_ok=True)
    fname = f"save/DPcnnlstm_e{num_epochs}_epsilon{epsilon}_{uuid.uuid1().hex}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(epoch_results, f)

    return epoch_results
