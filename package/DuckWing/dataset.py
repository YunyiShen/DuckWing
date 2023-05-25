import torch
from torch.utils.data import Dataset
import json
from PIL import Image

class SpeciesIndex():
    def __init__(self, species_list):
        self.species_list = species_list
    def __call__(self, species):
        return self.species_list.index(species)
    def getSpeciesName(self, index):
        return self.species_list[index]


class DuckWingPic(Dataset): 
    '''
    dataset class, take a json file, the list is a dictionary with Wing, Head, Belly, species
        items are wing image, head image, belly image, species label (in numbers)
    '''
    def __init__(self, data, sppindex, transform=None):
        
        self.data = data
        self.transform = transform
        self.sppindex = sppindex # a SpeciesIndex object

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data[idx][0] is None:
            wing = torch.randn(3, 384, 384) # random image
        else:
            wing = Image.open(self.data[idx][0]).convert('RGB')
        if self.data[idx][1] is None:
            head = torch.randn(3, 384, 384)
        else:
            head = Image.open(self.data[idx][1]).convert('RGB')
        if self.data[idx][2] is None:
            belly = torch.randn(3, 384, 384)
        else:
            belly = Image.open(self.data[idx][2]).convert('RGB')
        species = torch.tensor( [self.data[idx][3]])
        if self.transform:
            wing = self.transform(wing)
            head = self.transform(head)
            belly = self.transform(belly)
        return wing, head, belly, species

