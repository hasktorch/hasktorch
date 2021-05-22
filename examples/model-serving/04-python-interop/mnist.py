import torch as T
import torch.nn as nn
from torchvision import datasets, transforms
from collections import namedtuple

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = T.relu(self.conv1(x))
        x = T.relu(self.conv2(x))
        x = self.max_pool2d(x, 2)
        x = T.flatten(x, 1)
        x = T.relu(self.fc1(x))
        x = T.relu(x)
        x = T.log_softmax(T.fc2(x), dim=1)
        return x

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('data/', train=True, download=True, transform = transform)
    test_data = datasets.MNIST('data/', train=False, download=True, transform = transform)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = T.utils.data.DataLoader(test_data, batch_size=batch_size)
    return namedtuple("MNIST", "train test train_loader test_loader")(train_data, test_data, train_loader, test_loader)

if __name__ == "__main__":
    batch_size = 64
    device = "cpu"
    mnist = load_data(batch_size)


