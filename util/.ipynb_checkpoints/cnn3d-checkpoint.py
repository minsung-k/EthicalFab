import torch
import torch.nn as nn
import torch.nn.functional as F



class cnn3d(nn.Module):
    def __init__(self, num_classes):
        super(cnn3d, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.drop = nn.Dropout(p=0.3)
        
        # Adjust based on input size: (196, 32, 32) -> (D, 4, 4) after pooling twice
        self.fc_input_size = 2048  # Adjust based on max_seq and pooling
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # Change to [batch_size, 1, max_seq, 32, 32]
        x = self.pool(F.relu(self.bn(self.conv1(x))))
        x = self.pool(F.relu(self.bn(self.conv2(x))))
        x = self.pool(F.relu(self.bn(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.drop(x)
        x = self.fc(x)  # Final output
        return x


class cnn3d2(nn.Module):
    def __init__(self, num_classes):
        super(cnn3d, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=16)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=32)
        self.drop = nn.Dropout(p=0.3)
        
        # Adjust based on input size: (196, 32, 32) -> (D, 16, 16) after pooling twice
        self.fc_input_size = 32 * 196 * 8 * 8  # Adjust based on max_seq and pooling
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # Change to [batch_size, 1, max_seq, 32, 32]
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.drop(x)
        x = self.fc(x)  # Final output
        return x
