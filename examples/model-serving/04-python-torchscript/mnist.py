from typing import NamedTuple
import torch as T
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, datasets
from torch.utils.data.dataloader import DataLoader
from collections import namedtuple
from torch.nn.functional import nll_loss

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = T.relu(self.conv1(x))
        x = T.relu(self.conv2(x))
        x = T.max_pool2d(x, 2)
        x = T.flatten(x, 1)
        x = T.relu(self.fc1(x))
        x = T.relu(x)
        x = T.log_softmax(self.fc2(x), dim=1)
        return x

class MNIST(NamedTuple):
    train: datasets.mnist.MNIST
    test: datasets.mnist.MNIST
    train_loader: DataLoader
    test_loader: DataLoader


def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('data/', train=True, download=True, transform = transform)
    test_data = datasets.MNIST('data/', train=False, download=True, transform = transform)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = T.utils.data.DataLoader(test_data, batch_size=batch_size)
    return namedtuple("MNIST", "train test train_loader test_loader")(train_data, test_data, train_loader, test_loader)

def train(model: nn.Module, mnist_data: MNIST, device='cpu'):
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(mnist_data.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("epoch: {} | batch: {} loss {:.6f}".format(epoch,batch_idx, loss.item()))
    return model


if __name__ == "__main__":
    batch_size = 256
    device = "cpu"
    # device = T.device("cuda:0")
    mnist_data = load_data(batch_size)
    model = CNN().to(device)
    trained = train(model, mnist_data, device)
    T.save(dict(trained.state_dict()), "mnist.dict.pt")

    example_tensor, example_class = mnist_data.test[0]
    example_tensor = example_tensor.reshape([1, 1, 28, 28])
    # example_dict = {'example': example_tensor} # Without dummy throws "Exception: Unknown opcode for unpickling at 0x73: ..."
    example_dict = {'example': example_tensor, 'dummy': None} # C++ throws an exception for a 1-element dict

    print(model(example_tensor))

    T.save(example_dict , 'mnist.example.pt')

    traced = T.jit.trace(trained, example_inputs = example_tensor.to(device))
    print(type(traced))
    print(traced.graph)

    T.jit.save(traced, "mnist.ts.pt")

