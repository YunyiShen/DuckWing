import torch
import timm
import torch.nn as nn
from tqdm import tqdm


class wingMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(wingMLP, self).__init__()
        self.fc1 = nn.Linear(3 * embedding_dim, hidden_dim) # three images per sample
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_backbone(backbone, pretrained = True):
    model = timm.create_model(backbone, 
                      pretrained=pretrained,
                      num_classes=0)
    if(pretrained):
            for param in self.backbone.parameters():
                param.requires_grad = False
    return model
    

class DuckWing(nn.Module):
    def __init__(self, backbone, head, 
                 name = "my_model"):
        super(DuckWing, self).__init__()
        self.backbone = backbone
        self.head = head
        self.name = name

        # Freeze the parameters of the backbone if use pretrained
        
    
    def forward(self, wing, head, belly):
        wing = self.backbone(wing) 
        head = self.backbone(head)
        belly = self.backbone(belly)
        # difference between the two image embeddings, one way to make sure symmetry
        x = torch.cat((wing, head, belly), dim = 1)
        x = self.head(x)
        return x