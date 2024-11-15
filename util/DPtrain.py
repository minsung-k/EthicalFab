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

import uuid

from util import cnnlstm, visiontransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
def DPcnn3d(model, train_loader, val_loader, criterion, optimizer, num_epochs, epsilon):
    epoch_results = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        num_videos = 0

        # Training phase
        for frames, labels, mask, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()
            
            frames = frames.to(device)
            labels = labels.to(device')
            mask = mask.to(device)
            position_intervals = position_intervals.to(device)

            # Forward pass
            outputs = model(frames)  # Shape: [batch_size, num_classes]
            
            # Compute loss
            expanded_labels = labels.unsqueeze(1).expand(-1, frames.size(1))  # Match label shape to frames
            losses = criterion(outputs, expanded_labels[:, 0])  # Per-sample loss

            # Apply mask to losses and sum only non-padded frames
            masked_losses = losses * mask[:, 0]  # Shape: [batch_size]
            final_loss = masked_losses.sum() / mask[:, 0].sum()  # Mean loss over real frames
            total_loss += final_loss.item()

            # Backward pass and optimization
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

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
    with open(f'save/DP3dcnn_e{num_epochs}_epsilon{epsilon}_{uuid.uuid1().hex}.pkl', 'wb') as f:
        pickle.dump(epoch_results, f)

    return epoch_results

#------------------------------------------------------------------------------

def DPcnnlstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, epsilon):
    epoch_results = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        num_videos = 0  # Count number of videos processed
    
        # Training phase
        for inputs, targets, masks, max_seq, position_intervals in train_loader:
            optimizer.zero_grad()
            
            # Move data to GPU and ensure they are in float precision
            inputs, targets, masks = inputs.to(device).float(), targets.to(device), masks.to(device).float()
            
            # Forward pass
            outputs = model(inputs)  # Shape: [batch_size, seq_length, num_classes]
            batch_size, seq_length, num_classes = outputs.size()
            
            # Repeat the targets across the sequence length
            targets_expanded = targets.unsqueeze(1).expand(-1, seq_length)  # Shape: [batch_size, seq_length]
            targets_expanded = targets_expanded.contiguous().view(-1)  # Flatten to [batch_size * seq_length]
            
            # Compute loss
            loss = criterion(outputs.view(-1, num_classes), targets_expanded)  # Now targets are the same shape as flattened outputs
            loss = loss.view(batch_size, seq_length)  # Reshape loss to [batch_size, seq_length]
            
            # Apply mask to ignore padding
            loss = loss * masks  # Masks is [batch_size, seq_length] with 1s for valid frames and 0s for padding
            
            # Select non-zero values in masked loss (valid entries only)
            valid_loss = loss[masks != 0]  # This flattens the loss to remove padding
            
            # Calculate average loss for valid entries only if there are any valid values
            if valid_loss.numel() > 0:
                valid_loss = valid_loss.mean()  # Average valid entries
            else:
                valid_loss = torch.tensor(0.0, device=loss.device)  # Handle edge case with no valid entries
                
            # Backward pass
            valid_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
    
            # Accumulate total loss
            total_loss += valid_loss.item()
            
            # Calculate accuracy
            average_output = outputs.mean(dim=1)  # Shape: [batch_size, num_classes]
            
            # Now make the prediction for each sample
            _, predicted = torch.max(average_output, 1)  # Shape: [batch_size]
            
            # Now calculate the number of correct predictions
            total_correct += (predicted == targets).sum().item()
            num_videos += targets.size(0)  # Increment video count by batch size

        # Average loss and accuracy for training
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / num_videos
    
        # Evaluation phase
        # Validation phase with interval accuracy tracking
        model.eval()
        val_loss = 0
        val_correct = 0
        val_videos = 0
        interval_correct = {i: 0 for i in range(10)}
        interval_counts = {i: 0 for i in range(10)}
        
        with torch.no_grad():
            for inputs, targets, masks, max_seq, position_intervals in val_loader:
                inputs, targets, masks = inputs.to(device).float(), targets.to(device), masks.to(device).float()
                
                outputs = model(inputs)
                batch_size, seq_length, num_classes = outputs.size()
    
                # Repeat the targets across the sequence length
                targets_expanded = targets.unsqueeze(1).expand(-1, seq_length)  # Shape: [batch_size, seq_length]
                targets_expanded = targets_expanded.contiguous().view(-1)  # Flatten to [batch_size * seq_length]
                
                # Compute loss
                loss = criterion(outputs.view(-1, num_classes), targets_expanded)  # Now targets are the same shape as flattened outputs
                loss = loss.view(batch_size, seq_length)  # Reshape loss to [batch_size, seq_length]
                
                # Apply mask to ignore padding
                loss = loss * masks  # Masks is [batch_size, seq_length] with 1s for valid frames and 0s for padding
    
                val_loss += loss[masks != 0].mean() # Accumulate valid loss for each batch
    
                average_output = outputs.mean(dim=1)  # Shape: [batch_size, num_classes]
            
                # Now make the prediction for each sample
                _, predicted = torch.max(average_output, 1)  # Shape: [batch_size]
    
                val_correct += (predicted == targets).sum().item()
                val_videos += targets.size(0)
    
                # Interval accuracy
                for i in range(10):
                    interval_indices = (position_intervals == i)
                    interval_correct[i] += (predicted[interval_indices] == targets[interval_indices]).sum().item()
                    interval_counts[i] += interval_indices.sum().item()

    
        # Calculate interval accuracy
        interval_accuracy = {i: (interval_correct[i] / interval_counts[i] if interval_counts[i] > 0 else 0.0) for i in range(10)}
    
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
        
        # Print results
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print("Interval Accuracies:", interval_accuracy)
    
    # Save results
    with open(f'save/DPcnnlstm_e{num_epochs}_epsilon{epsilon}_{uuid.uuid1().hex}.pkl', 'wb') as f:
        pickle.dump(epoch_results, f)
    
    return epoch_results