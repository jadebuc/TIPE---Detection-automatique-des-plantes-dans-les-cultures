import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
import torch

from cnn import CNN

import os


# Transformation des images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Chargement des données de test
test_dataset = datasets.ImageFolder('datasets/test/images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)  # pas de mélange pour évaluer

#Choix du support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modele = CNN().to(device)  # on envoie le modèle sur le GPU ou CPU
modele.load_state_dict(torch.load("cnn.pth",map_location=device))
# Définition de la fonction de coût et de l'optimiseur
critere = nn.CrossEntropyLoss()
optimiseur = optim.Adam(modele.parameters(), lr=0.001)

# évaluation
correct = 0
total = 0
modele.eval()  

with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = modele(images)
        _, predi = torch.max(outputs.data, 1)  
        total += labels.size(0)
        correct += (predi == labels).sum().item()
print(test_dataset.class_to_idx)


print(f"Accuracy: {100 * correct / total:.2f}%")


torch.save(modele.state_dict(), "cnn.pth")
