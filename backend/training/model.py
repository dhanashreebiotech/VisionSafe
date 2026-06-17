
import torch
import torch.nn as nn

class ActivityClassifier(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=64, num_classes=4, num_layers=2):
        super(ActivityClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: [Batch, Seq, Feat]
        x = x.permute(0, 2, 1) # [B, Feat, Seq]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1) # [B, Seq, Feat]
        
        output, _ = self.lstm(x)
        
        # Last Step
        x = output[:, -1, :] 
        
        x = self.dropout2(x)
        x = self.fc(x)
        return x
