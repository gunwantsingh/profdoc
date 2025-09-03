import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # import the syft library

hook = sy.TorchHook(torch)  # attach the pytorch hook
joe = sy.VirtualWorker(hook, id="joe")  #  remote worker joe
jane = sy.VirtualWorker(hook, id="jane")  #  remote worker  jane

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

train_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=True, download=True, transform=transform)
test_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=False, download=True, transform=transform)

federated_train_loader = sy.FederatedDataLoader(
    train_set.federate((joe, jane)), batch_size=64, shuffle=True) # the federate() method splits the data within the workers

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
model = Net()
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epoch = 10 
for epoch in range(n_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location) # send the model to the client device where the data is present
        optimizer.zero_grad()         # training the model
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # get back the improved model
        if batch_idx % 100 == 0: 
            loss = loss.get() # get back the loss
            print('Training Epoch: {:2d} [{:5d}/{:5d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * 64,
                len(federated_train_loader) * 64,
                100. * batch_idx / len(federated_train_loader), loss.item()))

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(
            output, target, reduction='sum').item()
        pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss,correct,
    len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
