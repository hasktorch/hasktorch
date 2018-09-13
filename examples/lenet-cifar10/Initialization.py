import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        return F.log_softmax(x)


def train(net, nepochs, optimizer, criterion, trainloader, report_at, lr):
    for epoch in range(nepochs):  # loop over the dataset multiple times
        running_loss = 0.0
        seen = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            for param in net.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for param in net.parameters():
                param.data.add_((-lr), param.grad.data)

            # print(net.conv1.weight)
            # print statistics
            running_loss += loss.item()
            if i % report_at == (report_at-1):    # print every mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / report_at))
                running_loss = 0.0
            # if seen > 2000:
            #     break
            seen += 1

    print('Finished Training')

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))



def report(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on %d test images: %d %%' % (total, 100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def main(batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    numpy.random.seed(14)
    trainset = list(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform))
    numpy.random.shuffle(trainset)
    trainloader = torch.utils.data.DataLoader(trainset[:8000], batch_size=batch_size, shuffle=True, num_workers=2)

    testset = list(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform))
    numpy.random.shuffle(testset)
    testloader = torch.utils.data.DataLoader(testset[:8000], batch_size=batch_size, shuffle=False, num_workers=2)

    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    for p in net.parameters():
        p.requires_grad_(True)

    report(net, testloader)
    train(net, 3, optimizer, criterion, trainloader, 100, lr=0.001)
    report(net, trainloader)

if __name__ == "__main__":
    main()
