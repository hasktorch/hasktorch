import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy

numpy.set_printoptions(threshold=numpy.nan, precision=4, suppress=True, linewidth=1000)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# fclayer = 16 * 5 * 5
fclayer = 3 * 32 * 32
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(fclayer, 10)
        # self.fc1 = nn.Linear(fclayer, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # n = x.data.numpy()
        # print(n)
        # print((n <  1000000000).any())
        # print((n > -1000000000).any())

        # raise RuntimeError("")
        # x = self.pool(x) # F.relu(x))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, fclayer)
        x = (self.fc1(x))
        print(x)
        # x = (self.fc2(x))
        # x = self.fc3(x)
        return F.softmax(x, dim=1)


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
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            # print(loss)
            # print(loss.grad)
            loss.backward()

            # layers = list(net.parameters())
            # lst = layers[-1]
            # print(lst.grad)
            # print(loss)
            for param in net.parameters():
                # if len(param.grad.shape) == 2 and [param.grad.shape[0], param.grad.shape[1]] == [10, 84]:
                #   print(param.grad.data)

                param.data.add_((-lr), param.grad.data)

            # if (i+1) == 2:
            #   raise RuntimeError("!!!")

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
    trainloader = torch.utils.data.DataLoader(trainset[:400], batch_size=batch_size, shuffle=True, num_workers=2)

    # testset = list(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform))
    # numpy.random.shuffle(testset)
    # testloader = torch.utils.data.DataLoader(testset[:8000], batch_size=batch_size, shuffle=False, num_workers=2)

    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    for p in net.parameters():
        p.requires_grad_(True)

    # report(net, testloader)
    train(net, 50, None, criterion, trainloader, 100, lr=0.01)
    report(net, trainloader)

if __name__ == "__main__":
    main()
