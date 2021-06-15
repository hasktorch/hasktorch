import torch
import torch.nn as nn
import torch.optim as optim

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = nn.Linear(2, 1)

model = Simple()

print("state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(dict(model.state_dict()), 'simple.dict.pt')
