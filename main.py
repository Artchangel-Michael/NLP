from torch import nn
import torch
import numpy as np

#Задание 1
net_seq = nn.Sequential(
    nn.Linear(in_features=3, out_features=5),
    nn.Sigmoid(),
    nn.Linear(in_features=5, out_features=2),
)
if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'
net_seq.to(device)
lin1 = nn.Linear(in_features=3, out_features = 5)
lin2 = nn.Linear(in_features=5, out_features = 2)
print(lin1.weight, lin1.bias)
print(lin2.weight, lin2.bias)

#Общее количество обучаемых параметров = 32
#В первом слое количество весов 5*3, а смещений 5
#Во втором слое количество весов 2*5, а смещений 2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        return F.sigmoid(self.fc2(x))

net_model = Model()
net_model.to(device)
net_model
#Количество обучаемых параметров такое же, как и у класса net_seq

#Задание 2
class NeuronNOT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.heaviside(self.fc(x), torch.tensor([0.0]))
    
neuronNOT = NeuronNOT()
neuronNOT.fc.weight, neuronNOT.fc.bias
neuronNOT.fc.weight.data = torch.tensor([[-0.5]])
neuronNOT.fc.bias.data = torch.tensor([0.5])
x0 = torch.tensor([[0.],
                 [1.]])
print(neuronNOT(x0))



#Задание 3
class NeuronAND(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(2, 1)

  def forward(self, x):
    return torch.heaviside(self.fc(x), torch.tensor([0.0]))
  
neuronAND = NeuronAND()
neuronAND.fc.weight, neuronAND.fc.bias

neuronAND.fc.weight.data = torch.tensor([[0.5, 0.5]])
neuronAND.fc.bias.data = torch.tensor([-0.9])

x = torch.tensor([[0.0, 0.0],
                 [0.0, 1.0],
                 [1.0, 0.0],
                 [1.0, 1.0]])
print(neuronAND(x))


#Задание 4
class NeuronOR(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(2, 1)

  def forward(self, x):
    return torch.heaviside(self.fc(x), torch.tensor([0.0]))
neuronOR = NeuronOR()
neuronOR.fc.weight, neuronOR.fc.bias

neuronOR.fc.weight.data = torch.tensor([[0.5, 0.5]])
neuronOR.fc.bias.data = torch.tensor([-0.4])

print(neuronOR(x))

#Задание 5
class NeuronXOR(torch.nn.Module):
    def __init__(self):  
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

        self.hidden.weight.data = torch.tensor([[0.5, 0.5],   # OR
                                                [-0.5, -0.5]]) # NOT AND
        self.hidden.bias.data = torch.tensor([-0.4, 0.9])

        self.output.weight.data = torch.tensor([[0.5, 0.5]])  # AND
        self.output.bias.data = torch.tensor([-0.9])

    def forward(self, x):
        h = torch.heaviside(self.hidden(x), torch.tensor([0.0]))
        y = torch.heaviside(self.output(h), torch.tensor([0.0]))
        return y

neuronXOR = NeuronXOR()
print(neuronXOR(x))


#Задание 6
class Neuron(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(2, 1)

  def forward(self, x):
    return torch.heaviside(self.fc(x), torch.tensor([0.0]))

# x = torch.tensor([[0.0, 0.0]])
neuron = Neuron()
neuron.fc.weight, neuron.fc.bias
neuron.fc.weight.data = torch.tensor([[-0.5, -0.5]])
neuron.fc.bias.data = torch.tensor([1.0])
n1 = neuron(x)
neuron.fc.weight.data = torch.tensor([[0.5, 0.5]])
neuron.fc.bias.data = torch.tensor([-0.4])
n2= neuron(x)
neuron.fc.weight.data = torch.tensor([[0.5, 0.5]])
neuron.fc.bias.data = torch.tensor([-0.9])
output_ns = torch.stack((n1, n2), dim = 1)
inp_n3 = torch.squeeze(output_ns, dim = 2)
print(neuron(inp_n3))