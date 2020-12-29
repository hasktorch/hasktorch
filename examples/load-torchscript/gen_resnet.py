# Original source   https://pytorch.org/tutorials/advanced/cpp_export.html
# The artifact of this model is in https://github.com/hasktorch/libtorch-binary-for-ci/releases/tag/1.4.0.

import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(True)
model = model.eval()

# An example input you would normally provide to your model's forward() method.
#example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("resnet_model.pt")

script_module = torch.jit.script(model)
script_module.save("resnet_model.pt")
