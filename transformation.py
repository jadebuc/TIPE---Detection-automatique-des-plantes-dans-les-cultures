from torchvision import datasets,transforms #dtaset = charge les jeux d'images, transforms = transformations sur images
from torch.utils.data import DataLoader #charge par petit lots (batchs), mélange les données (shuffle), charge efficacement
transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
#transforme chaque image, compose=compose transfo, totensor = convertit en un tenseur PyTorch entre 0 et 1 et la met au format [canal,hauteur,largeur]

train_dataset= datasets.ImageFolder('datasets/train',transform=transform)
test_dataset=datasets.ImageFolder('datasets/test',transform=transform)
train_loader=DataLoader(train_dataset, batch_size=2, shuffle=True)#mélange aide a généraliser
test_loader=DataLoader(test_dataset, batch_size=2, shuffle=False)#pas mlanger pour le test pour comparer

