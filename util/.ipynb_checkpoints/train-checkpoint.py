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

from util import cnnlstm, visiontransformer

import uuid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def cnn3d(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    epoch_results = []
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        num_videos = 0

        # Training phase
        for frames, labels, mask, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()

            frames = frames.to(device)  # Shape: [batch_size, max_seq, 1, 32, 32]
            labels = labels.to(device)  # Shape: [batch_size]
            mask = mask.to(device)      # Shape: [batch_size, max_seq]
            position_intervals = position_intervals.to(device)  # Shape: [batch_size]

            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(frames)  # Shape: [batch_size, num_classes]
                
                # Compute loss for each batch based on real frames only
                expanded_labels = labels.unsqueeze(1).expand(-1, frames.size(1))  # Match label shape to frames
                losses = criterion(outputs, expanded_labels[:, 0])  # Per-sample loss

                # Apply mask to losses and sum only non-padded frames
                masked_losses = losses * mask[:, 0]  # Shape: [batch_size]
                final_loss = masked_losses.sum() / mask[:, 0].sum()  # Mean loss over real frames
                total_loss += final_loss.item()

            if torch.isfinite(final_loss):
                # Scale the gradients and backpropagation
                scaler.scale(final_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
        val_loss = 0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}  # Accuracy per interval
        interval_counts = {i: 0 for i in range(10)}  # Counts per interval

        with torch.no_grad():
            for frames, labels, mask, max_seq, position_intervals in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                position_intervals = position_intervals.to(device)

                outputs = model(frames)
                expanded_labels = labels.unsqueeze(1).expand(-1, frames.size(1))
                losses = criterion(outputs, expanded_labels[:, 0])

                masked_losses = losses * mask[:, 0]
                final_val_loss = masked_losses.sum() / mask[:, 0].sum()
                val_loss += final_val_loss.item()

                _, predicted = torch.max(outputs, 1)  # Get predictions
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
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_videos

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

def cnnlstm(model, train_loader, val_loader, criterion, optimizer, num_epochs):
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

