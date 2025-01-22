import torch
import torch.nn as nn
import torch.nn.functional as F


class Semantic_network(nn.Module):
    def __init__(self, drop_rate=0.2, filters1=271, filters2=271):
        super(Semantic_network, self).__init__()
        conv_weights = torch.rand((256, 4, 5))
        conv_weights = torch.clamp(conv_weights, 0, 1)
        self.conv1d_layer = nn.Conv1d(in_channels=4, out_channels=256, padding=2, kernel_size=5)
        self.conv1d_layer.weight.data = conv_weights

        # Define layers
        self.conv1 = nn.Conv1d(in_channels=271, out_channels=filters1, kernel_size=5, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(filters1)
        self.conv2 = nn.Conv1d(in_channels=filters1, out_channels=filters2, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=1)
        self.dropout1 = nn.Dropout(drop_rate)

        self.conv3 = nn.Conv1d(in_channels=271, out_channels=filters1, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(filters1)
        self.conv4 = nn.Conv1d(in_channels=filters1, out_channels=filters2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=1)
        self.dropout2 = nn.Dropout(drop_rate)

        self.lstm1 = nn.LSTM(input_size=filters2, hidden_size=271, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=filters2, hidden_size=271, bidirectional=False, batch_first=True)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, x):

        x=x.permute(0,2,1)
        out1 = F.tanh(self.conv1(x))
        out1 = self.batchnorm1(out1)

        out1 = self.dropout1(out1)

        out1 = F.relu(self.conv2(out1))
        out1 = self.maxpool1(out1)
        out1 = self.dropout1(out1)

        # Bidirectional LSTM
        outx, _ = self.lstm1(out1.permute(0, 2, 1))
        outy, _ = self.lstm2(out1.permute(0, 2, 1))



        out =(outx+outy)/2.0

        out = self.dropout3(out)

        return out


