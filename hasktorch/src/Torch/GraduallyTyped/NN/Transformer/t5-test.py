import torch
from transformers import AutoTokenizer, T5Model

model = T5Model.from_pretrained('t5-small', torchscript=False)
model.eval()

batch_size = 1
encoder_seq_length = 100
decoder_seq_length = 4
d_model = 512

inputs_embeds = torch.ones([batch_size, encoder_seq_length, d_model], dtype=torch.float)
decoder_inputs_embeds = torch.ones([batch_size, decoder_seq_length, d_model], dtype=torch.float)
attention_mask = torch.ones([batch_size, encoder_seq_length], dtype=torch.float)
decoder_attention_mask = torch.ones([batch_size, decoder_seq_length], dtype=torch.float)

model_output = model.forward(
	inputs_embeds=inputs_embeds,
	decoder_inputs_embeds=decoder_inputs_embeds,
	attention_mask=attention_mask,
	decoder_attention_mask=decoder_attention_mask
)
print(model_output.last_hidden_state)

mask = torch.ones([batch_size, decoder_seq_length], dtype=torch.bool)
print(model.decoder.get_extended_attention_mask(mask, (batch_size, decoder_seq_length), mask.device))
