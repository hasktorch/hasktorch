import torch
import torch.nn as nn
import torch.optim as optim

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = nn.Linear(2, 1)

print("state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

model2 = Simple()

torch.save(model2, 'test2.pt')
torch.save(dict(model2.state_dict()), 'test2sd.pt')

# check = torch.load('test.pt')
