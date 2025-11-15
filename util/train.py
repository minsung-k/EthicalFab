import os
import cv2
import glob
from PIL import Image
import numpy as np
import pickle

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from opacus.layers import DPLSTM
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings
warnings.filterwarnings('ignore')

from util import cnnlstm

import uuid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def cnn3d(model, train_loader, val_loader, criterion, optimizer, num_epochs, device: torch.device | None = None):
    """
    Train a 3D CNN model.

    Parameters
    ----------
    model : torch.nn.Module
        The 3D CNN model to train. It will automatically be moved to the
        specified device.
    train_loader : DataLoader
        Dataloader yielding training batches.
    val_loader : DataLoader
        Dataloader yielding validation batches.
    criterion : callable
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimiser used to update model parameters.
    num_epochs : int
        Number of epochs to train.
    device : torch.device, optional
        Device on which to perform computations. If ``None`` the function
        defaults to CUDA when available, otherwise CPU.

    Returns
    -------
    list[dict]
        A list of dictionaries containing metrics for each epoch.
    """
    # Select device once
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epoch_results: list[dict] = []
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))  # For mixed precision only on CUDA

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        num_videos = 0

        # Training phase
        for frames, labels, mask, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()

            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            # ``position_intervals`` is only used for metrics and does not
            # influence gradients. Keeping it on CPU is fine, but for
            # consistency we move it as well.
            position_intervals = position_intervals.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # Forward pass
                outputs = model(frames)

                # Compute loss for each batch based on real frames only.
                expanded_labels = labels.unsqueeze(1).expand(-1, frames.size(1))
                losses = criterion(outputs, expanded_labels[:, 0])

                # Apply mask to losses and sum only non-padded frames
                # Use the first time step's mask since all frames in the batch
                # correspond to the same number of valid frames.
                masked_losses = losses * mask[:, 0]
                final_loss = masked_losses.sum() / mask[:, 0].sum()
                total_loss += final_loss.item()

            if torch.isfinite(final_loss):
                scaler.scale(final_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            num_videos += labels.size(0)


        # Average loss and accuracy for training
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / num_videos

        # Validation phase with interval accuracy tracking
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}
        interval_counts = {i: 0 for i in range(10)}

        with torch.no_grad():
            for frames, labels, mask, max_seq, position_intervals in val_loader:
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                position_intervals = position_intervals.to(device, non_blocking=True)

                outputs = model(frames)
                expanded_labels = labels.unsqueeze(1).expand(-1, frames.size(1))
                losses = criterion(outputs, expanded_labels[:, 0])

                masked_losses = losses * mask[:, 0]
                final_val_loss = masked_losses.sum() / mask[:, 0].sum()
                val_loss += final_val_loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_videos += labels.size(0)

                # Calculate interval accuracy
                for i in range(10):
                    interval_indices = (position_intervals == i)
                    interval_correct[i] += (predicted[interval_indices] == labels[interval_indices]).sum().item()
                    interval_counts[i] += interval_indices.sum().item()

        # Calculate average interval accuracy
        interval_accuracy = {i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0)
                             for i in range(10)}

        # Average loss and accuracy for validation
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = val_correct / val_videos if val_videos > 0 else 0.0

        # Save epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'train_accuracy': train_accuracy,
            'avg_val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'interval_accuracy': interval_accuracy
        })


        # Print epoch and interval results
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        print("Interval Accuracies:", interval_accuracy)

    # Save epoch results to a file
    with open(f'save/3dcnn_e{num_epochs}_{uuid.uuid1().hex}.pkl', 'wb') as f:
        pickle.dump(epoch_results, f)

    return epoch_results




#------------------------------------------------------------------------------

def cnnlstm_old(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # List to store epoch results
    epoch_results = []
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        num_videos = 0  # Count number of videos processed

        # Training phase
        for inputs, targets, masks, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs, masks)  # Shape: (batch_size, num_classes)
                loss = criterion(outputs, targets).mean()
                total_loss += loss.item()

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the predicted classes
            total_correct += (predicted == targets).sum().item()
            num_videos += targets.size(0)

        # Average loss and accuracy for training
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / num_videos

        # Validation phase with interval accuracy tracking
        model.eval()
        val_loss = 0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}  # Accuracy per interval
        interval_counts = {i: 0 for i in range(10)}  # Counts per interval
        
        with torch.no_grad():
            for inputs, targets, masks, max_seq, position_intervals in val_loader:
                inputs, targets, position_intervals = inputs.to(device), targets.to(device), position_intervals.to(device)

                # Forward pass
                outputs = model(inputs, masks)
                loss = criterion(outputs, targets).mean()
                val_loss += loss.item()
        
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_videos += targets.size(0)

                

                # Calculate interval accuracy
                for i in range(10):
                    interval_indices = (position_intervals == i)
                    interval_correct[i] += (predicted[interval_indices] == targets[interval_indices]).sum().item()
                    interval_counts[i] += interval_indices.sum().item()

        # Calculate average interval accuracy
        interval_accuracy = {i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0)
                             for i in range(10)}

        # Average loss and accuracy for validation
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = val_correct / val_videos if val_videos > 0 else 0.0
        
        # Save epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'train_accuracy': train_accuracy,
            'avg_val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'interval_accuracy': interval_accuracy
        })

        
        # Print epoch and interval results
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        print("Interval Accuracies:", interval_accuracy)

    # Save epoch results to a file
    with open(f'save/cnnlstm_e{num_epochs}_{uuid.uuid1().hex}.pkl', 'wb') as f:
        pickle.dump(epoch_results, f)

    return epoch_results


# -----------------------------------------------------------------------------
# Revised implementation of cnnlstm
def cnnlstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device: torch.device | None = None):
    """
    Train a CNN‑LSTM model on the given data loaders.

    This function mirrors the original training loop but introduces several
    improvements:

    * The model is moved to the appropriate device once at the beginning.
    * Automatic mixed precision is enabled when running on CUDA.
    * Data is transferred to the device using non‑blocking copies where
      possible.

    Parameters
    ----------
    model : torch.nn.Module
        The network to train. Must implement ``forward(inputs, masks)``.
    train_loader : torch.utils.data.DataLoader
        Loader providing training batches.
    val_loader : torch.utils.data.DataLoader
        Loader providing validation batches.
    criterion : callable
        Loss function returning a per-sample or scalar loss.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    num_epochs : int
        Number of epochs to train for.
    device : torch.device, optional
        Target device. Defaults to ``cuda`` if available, otherwise ``cpu``.

    Returns
    -------
    list of dict
        A list of dictionaries capturing statistics for each epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the desired device
    model = model.to(device)

    epoch_results: list[dict] = []
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        num_videos = 0

        for inputs, targets, masks, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs, masks)
                # Some privacy mechanisms return per-sample losses. Reduce to scalar.
                loss = criterion(outputs, targets).mean()
                total_loss += loss.item()

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            num_videos += targets.size(0)

        # Training statistics
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / num_videos

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_videos = 0
        interval_correct: dict[int, int] = {i: 0 for i in range(10)}
        interval_counts: dict[int, int] = {i: 0 for i in range(10)}

        with torch.no_grad():
            for inputs, targets, masks, max_seq, position_intervals in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                position_intervals = position_intervals.to(device, non_blocking=True)

                outputs = model(inputs, masks)
                v_loss = criterion(outputs, targets).mean()
                val_loss += v_loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_videos += targets.size(0)

                for i in range(10):
                    idxs = (position_intervals == i)
                    # Only accumulate when there are samples for this interval
                    if idxs.any():
                        interval_correct[i] += (predicted[idxs] == targets[idxs]).sum().item()
                        interval_counts[i] += idxs.sum().item()

        # Compute per-interval accuracy
        interval_accuracy = {i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0)
                             for i in range(10)}
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = val_correct / val_videos if val_videos > 0 else 0.0

        epoch_results.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'train_accuracy': train_accuracy,
            'avg_val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'interval_accuracy': interval_accuracy
        })

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print("Interval Accuracies:", interval_accuracy)

    # Persist metrics to disk
    with open(f'save/cnnlstm_e{num_epochs}_{uuid.uuid1().hex}.pkl', 'wb') as f:
        pickle.dump(epoch_results, f)
    return epoch_results

