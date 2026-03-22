import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn import CNN
import random
import numpy as np

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Transformations pour les images d'entraînement
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    
     #Ajout de ColorJitter
    #transforms.ColorJitter(
     #   brightness=0.1,   # variations de luminosité
      #  contrast=0.1,     # variations de contraste
       # saturation=0.1,   # variations de saturation
        #hue=0.02          # variations de teinte
    #),

    #Ajout de flou gaussien (aléatoirement)
    #transforms.RandomApply([
     #   transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    #], p=0.5), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Chargement du dataset
train_dataset = datasets.ImageFolder('datasets/train/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Utilisation du GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = CNN().to(device)
criterion = nn.CrossEntropyLoss() #fct de perte = mesure l'erreur entre ce que le modèle prédit et la bonne reponse
optimizer = optim.Adam(model.parameters(), lr=0.001) #optimiseur =ajuste les poids du réseau pour réduire l'erreur

train_losses = []
train_accuracies = []

# Boucle d'entraînement
for epoch in range(10):
    model.train()
    total_loss = 0.0
    nb_bonnes=0
    total=0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() #reinitialisation du gradient
        outputs = model(images) #passe les images dans le réseau
        loss = criterion(outputs, labels) #calcul de l'erreur
        total_loss+=loss.item()
        loss.backward() #calcul du gradient des poids
        optimizer.step() #mise à jour des poids

        _, predicted = torch.max(outputs, 1)
        nb_bonnes += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calcul et stockage des valeurs pour les courbes
    moyenne_loss = total_loss / len(train_loader)
    precision = 100 * nb_bonnes / total
    train_losses.append(moyenne_loss)
    train_accuracies.append(precision)

    print(f"Epoch {epoch+1}/10, Loss: {moyenne_loss:.4f}, Accuracy: {precision:.2f}%")
    
    """total_loss += loss.item()

    print(f"Epoch {epoch + 1}/30, Loss: {total_loss / len(train_loader):.4f}")"""

# Sauvegarde des poids du modèle
torch.save(model.state_dict(), "cnn.pth")

