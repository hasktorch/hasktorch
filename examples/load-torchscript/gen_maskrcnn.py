# Original source   https://pytorch.org/tutorials/advanced/cpp_export.html

import torch
import torchvision

# An instance of your model.
model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(3, 300, 400)
model.eval()
# tracing does not work for maskrcnn
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("traced_maskrcnn_model.pt")

script_module = torch.jit.script(model)
script_module.save("maskrcnn_model.pt")
#predictions = model(example.cpu())

#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#model.eval()
#x = torch.rand(1, 3, 300, 400)
(losses,detections) = script_module([example])
print(losses)
print(detections)
#print(predictions)
