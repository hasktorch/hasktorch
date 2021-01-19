import torch
from transformers import AutoTokenizer, T5Model

class T5Small(torch.nn.Module):
    def __init__(self):
        super(T5Small, self).__init__()
        self.model = T5Model.from_pretrained('t5-small', torchscript=True)

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

# model = T5Small()
# model.eval()

# tokenizer = AutoTokenizer.from_pretrained('t5-small')
# input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
# decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

# traced_model = torch.jit.trace(model, [input_ids, decoder_input_ids])
# traced_model.save("t5-small.pt")

model = T5Model.from_pretrained('t5-small', torchscript=False)
d = dict(model.state_dict())
for k, v in d.items():
    print("{}: {}".format(k, v.shape))

torch.save(d, "t5-small.pt", _use_new_zipfile_serialization=True)