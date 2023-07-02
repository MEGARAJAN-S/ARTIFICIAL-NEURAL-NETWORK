from torch import nn
#Building a neural network

class neural_network(nn.Module):

    def __init__(self):

        super(neural_network,self).__init__()
        #defining the input layer, hidden layer, output layer
        self.input = nn.Linear(784,128)
        self.hidden = nn.Linear(128,64)
        self.output = nn.Linear(64,10)

        #defining the activation function
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def Forward(self,signal):

        #Forwarding the input in the neural network
        signal = self.relu(self.input(signal))
        signal = self.relu(self.hidden(signal))
        signal = self.logsoftmax(self.output(signal))

        return signal

