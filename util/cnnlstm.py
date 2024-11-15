import os
import cv2
import glob
from PIL import Image
import numpy as np


import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets


from opacus.layers import DPLSTM
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

class cnnlstm(nn.Module):
    def __init__(self, num_classes, num_layers=1):
        super(cnnlstm, self).__init__()
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=16)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=32),
            nn.Dropout(p=0.3)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: (32, 16, 16)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32 * 8 * 8, hidden_size=256, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, mask):
        batch_size, seq_length, _, _, _ = x.size()
        
        # Reshape input for CNN
        x = x.view(-1, 1, 32, 32)  # Reshape to (batch_size * seq_length, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
    
        # Flatten the CNN output
        x = x.view(batch_size, seq_length, -1)  # Reshape to (batch_size, seq_length, features)
        
        # Calculate lengths from mask
        lengths = mask.sum(dim=1).cpu()  # Calculate the sum of mask values for each batch to get real frame counts
    
        # Filter out sequences with zero length
        non_zero_indices = lengths > 0
        if non_zero_indices.sum() == 0:
            return torch.zeros(batch_size, self.fc.out_features, device=x.device).to(x.dtype)  # Return a zero output if all sequences are padded
        
        x_non_zero = x[non_zero_indices]
        lengths_non_zero = lengths[non_zero_indices]
    
        # Pack the padded sequence for non-zero sequences
        packed_input = pack_padded_sequence(x_non_zero, lengths_non_zero, batch_first=True, enforce_sorted=False)
        
        # LSTM part
        packed_output, (hn, cn) = self.lstm(packed_input)
    
        # Unpack and select the last valid time step based on the length
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_time_step = lstm_out[torch.arange(lstm_out.size(0)), (lengths_non_zero - 1).long()]
    
        # Initialize output as zeros and fill only non-zero indices
        out = torch.zeros(batch_size, self.fc.out_features, device=x.device, dtype=x.dtype)  # Match the dtype here
        out[non_zero_indices] = self.fc(last_time_step).to(out.dtype)  # Convert to match output dtype if needed
        
        return out

#------------------------------------------------------------------------------

class DPcnnlstm(nn.Module):
    def __init__(self, num_classes, num_layers=1):
        super(DPcnnlstm, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=16),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=32),
            nn.Dropout(p=0.3)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: (32, 16, 16)
        
        # LSTM layer
        self.lstm = DPLSTM(32 * 8 * 8, 256, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()

        # List to hold the CNN outputs for each frame
        cnn_outputs = []

        # Loop over each frame in the sequence and apply CNN layers independently
        for i in range(seq_length):
            frame = x[:, i, :, :, :]  # Select the i-th frame from the sequence
            
            # Apply CNN layers to the frame
            frame = self.pool(F.relu(self.conv1(frame)))
            frame = self.pool(F.relu(self.conv2(frame)))
            
            # Flatten the CNN output and append to the list
            cnn_outputs.append(frame.view(batch_size, -1))  # Flatten each frame's output

        # Stack all CNN outputs to form a tensor [batch_size, seq_length, features]
        cnn_outputs = torch.stack(cnn_outputs, dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_outputs)  # LSTM output: [batch_size, seq_length, features]

        # Fully connected layer for final classification output
        out = self.fc(lstm_out)  # [batch_size, seq_length, num_classes]
    
        return out
#------------------------------------------------------------------------------