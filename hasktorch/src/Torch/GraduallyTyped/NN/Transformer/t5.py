import torch
from transformers import AutoTokenizer, T5Model, T5ForConditionalGeneration

# class T5Small(torch.nn.Module):
#     def __init__(self):
#         super(T5Small, self).__init__()
#         self.model = T5Model.from_pretrained('t5-small', torchscript=True)

#     def forward(self, input_ids, decoder_input_ids):
#         return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

# model = T5Small()
# model.eval()

# traced_model = torch.jit.trace(model, [input_ids, decoder_input_ids])
# traced_model.save("t5-small.pt")


def printExample(model):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    inputs = [
        "Studies have been shown that owning a dog is good for you",
        "Studies have been shown that owning a dog is good for you and you"]
    tokenized_inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    decoder_inputs = [
        "Studies show that",
        "Studies show that"]
    tokenized_decoder_inputs = tokenizer(decoder_inputs, padding="longest", return_tensors="pt")
    outputs = model(
        input_ids=tokenized_inputs.input_ids,
        attention_mask=tokenized_inputs.attention_mask,
        decoder_input_ids=tokenized_decoder_inputs.input_ids,
        decoder_attention_mask=tokenized_decoder_inputs.attention_mask)

    print(tokenized_inputs)
    print(tokenized_decoder_inputs)
    # print(outputs.last_hidden_state)
    print(outputs.logits)


# model = T5Model.from_pretrained('t5-small', torchscript=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small', torchscript=False)
model.eval()

printExample(model)

d = dict(model.state_dict())
# for k, v in d.items():
#     print("{}: {}".format(k, v.shape))
torch.save(d, "t5-small.pt", _use_new_zipfile_serialization=True)

model = T5ForConditionalGeneration.from_pretrained('t5-base', torchscript=False)
model.eval()

printExample(model)

d = dict(model.state_dict())
# for k, v in d.items():
#     print("{}: {}".format(k, v.shape))
torch.save(d, "t5-base.pt", _use_new_zipfile_serialization=True)
