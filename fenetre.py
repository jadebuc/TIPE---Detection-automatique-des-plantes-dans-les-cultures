import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from cnn import CNN

# choix du support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modele
model = CNN().to(device)
model.load_state_dict(torch.load("cnn.pth", map_location=device))
model.eval()

classes = {0: 'pommier', 1: 'myrtille'}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Fonction de chargement et de prédiction
def charger_image():
    chemin = filedialog.askopenfilename()
    if not chemin:
        return

    img = Image.open(chemin).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output.data, 1)
        resultat.set(f"Résultat : {classes[prediction.item()]}")

# Interface graphique
fenetre = tk.Tk()
fenetre.title("Pommier ou Cerisier ?")

bouton = tk.Button(fenetre, text="Choisir une image", command=charger_image)
bouton.pack(pady=10)

resultat = tk.StringVar()
label_resultat = tk.Label(fenetre, textvariable=resultat, font=("Arial", 20))
label_resultat.pack(pady=10)

fenetre.mainloop()

