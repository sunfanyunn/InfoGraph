import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal,xavier_uniform

# CNN Model
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, d, n_kernels, max_n_communities):
        super(CNN, self).__init__()
        self.max_n_communities = max_n_communities
        self.conv = nn.Conv3d(1, input_size, (1, 1, d), padding=0)
        self.fc1 = nn.Linear(input_size*n_kernels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.init_weights()

    def init_weights(self):
        xavier_uniform(self.conv.weight.data)
        xavier_normal(self.fc1.weight.data)
        xavier_normal(self.fc2.weight.data)

    def forward(self, x_in):
        out = F.relu(F.max_pool3d(self.conv(x_in), (1, self.max_n_communities,1)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
           