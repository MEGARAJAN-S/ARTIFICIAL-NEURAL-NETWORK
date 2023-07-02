from Program import ANN
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch import optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Defining the neural network

Brain = ANN.neural_network()
#Define the loss
criterion = nn.CrossEntropyLoss()

#Defining the optimizer
optimizer = optim.Adam(Brain.parameters(), lr=0.002)

#Defining the epochs
epochs = 20

#Training the model

for data in range(1,epochs+1):
    total_loss = 0
    for images,lables in trainloader:
        #Flating the image
        images = images.view(images.shape[0],-1)
        #Training pass
        optimizer.zero_grad()

        output = Brain.Forward(images)
        loss = criterion(output, lables)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    else:
        value = total_loss/len(trainloader)
        print("EPOCHS NO : {EPOCHS} Training loss : {Training_loss}".format(EPOCHS=data,Training_loss=value))
