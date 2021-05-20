import torch
import torch.nn as nn
import torch.optim as optim

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = nn.Linear(2, 1)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Test()

print("state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

model2 = Simple()

torch.save(model2, 'test2.pt')
torch.save(dict(model2.state_dict()), 'test2sd.pt')

foo = torch.load('test.pt')

torch.save(torch.tensor([1.0, 2.0, 3.0]), 'test3.pt')

torch.save([torch.tensor([1.0, 2.0, 3.0])], 'test4.pt')
