import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from sklearn.metrics import accuracy_score

# 1. Prepare data
# 2. Create model class
# 3. Create model class instance
# 4. Create loss instance
# 5. Create optimizer instance
# 6. Train model

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

X = train_dataset.data
Y = train_dataset.targets
X = X.view(X.shape[0], -1)
X = X.float()

# 2. Create model class
class FNN(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, layerOut):
        super(FNN, self).__init__()
        self.l1 = nn.Linear(input_dim, layer1_dim)
        self.l1_activ = nn.ReLU()
        self.l2 = nn.Linear(layer1_dim, layer2_dim)
        self.l2_activ = nn.ReLU()
        self.l3 = nn.Linear(layer2_dim, layerOut)
    
    def forward(self, x):
        z1 = self.l1(x)
        a1 = self.l1_activ(z1)
        z2 = self.l2(a1)
        a2 = self.l2_activ(z2)
        z3 = self.l3(a2)
        return z3

# 3. Create model class instance
input_dim = 28*28
layer1_dim = 80
layer2_dim = 40
layerOut = 10
model = FNN(input_dim, layer1_dim, layer2_dim, layerOut)

# 4. Create loss instance
loss = nn.CrossEntropyLoss()

# 5. Create optimizer instance
optim = torch.optim.SGD(model.parameters(), lr = 1e-2)

total_samples = len(Y)
batch_size = 100
epochs = 600

# 6. Train model
for epochCounter in range(epochs):
    epoch_loss = 0
    for batch_start in range(0, total_samples, batch_size):
        batch_end = batch_start + batch_size
        batch_X = X[batch_start:batch_end]
        batch_Y = Y[batch_start:batch_end]
        optim.zero_grad()
        batch_Z3 = model(batch_X)
        batch_loss = loss(batch_Z3, batch_Y)
        batch_loss.backward()
        optim.step() #adjusts weights?
        
        epoch_loss += batch_loss.item()
    print(epochCounter, epoch_loss)


true_labels = []
pred_labels = []
for batch_start in range(0, total_samples, batch_size):
    batch_end = batch_start + batch_size
    batch_X = X[batch_start:batch_end]
    batch_Y = Y[batch_start:batch_end]

    # forward pass
    cur_batch_O = model(batch_X)
    cur_batch_P = cur_batch_O.argmax(dim=1)
#     print(cur_batch_Y.shape, cur_batch_O.shape, cur_batch_P.shape)
    true_labels += batch_Y.numpy().tolist()
    pred_labels += cur_batch_P.detach().numpy().tolist()
#     break

print(accuracy_score(true_labels, pred_labels))