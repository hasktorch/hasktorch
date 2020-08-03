import torch
import torchvision

import torchvision.models as models

model = models.alexnet(pretrained=True)

example = torch.rand(1, 3, 224, 224)

model.eval()

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_pretrained_alexnet_model.pt")
