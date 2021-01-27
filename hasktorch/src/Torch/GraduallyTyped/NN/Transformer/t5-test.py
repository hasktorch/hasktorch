import torch
from transformers import AutoTokenizer, T5Model
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Attention
from transformers.models.t5.configuration_t5 import T5Config

model = T5Model.from_pretrained('t5-small', torchscript=False)
model.eval()

batch_size = 1
encoder_seq_length = 3
decoder_seq_length = 2
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

context_position = torch.arange(decoder_seq_length, dtype=torch.long)[:, None]
memory_position = torch.arange(decoder_seq_length, dtype=torch.long)[None, :]
relative_position = memory_position - context_position  # shape (query_length, key_length)
relative_position_bucket = model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket(
    relative_position,  # shape (query_length, key_length)
    bidirectional=False,
    num_buckets=model.config.relative_attention_num_buckets,
)
print(relative_position_bucket)

# embedding = torch.nn.Embedding(10, 3, padding_idx=None)
# input = torch.LongTensor([[0,2,0,5]])
# print(embedding(input))

layer_norm = T5LayerNorm(hidden_size=10, eps=1e-6)
print(layer_norm(torch.tensor([[13, 27, 14, 19, -512, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float)))

config = T5Config(
	    vocab_size=8,
	    d_model=3,
	    d_kv=2,
	    num_heads=1,
	    relative_attention_num_buckets=4,
	    dropout_rate=0.1,
	    layer_norm_epsilon=1e-6,
	    feed_forward_proj="relu",
	    is_decoder=False
	)
attention = T5Attention(config=config, has_relative_attention_bias=True)
attention.eval()
torch.nn.init.ones_(attention.q.weight)
torch.nn.init.ones_(attention.k.weight)
torch.nn.init.ones_(attention.v.weight)
torch.nn.init.ones_(attention.o.weight)
torch.arange(0, 4, out=attention.relative_attention_bias.weight)
hidden_states = torch.tensor([[[0, 1, 2], [-1, -2, -3], [7, -2, -3], [-1, 5, -3]]], dtype=torch.float)
mask = torch.zeros(1, 4, 3, dtype=torch.float)
key_value_states = torch.tensor([[[0, 0.5, 1], [-0.1, -0.2, -0.3], [-1, 0, 1]]], dtype=torch.float)
print(attention(hidden_states=hidden_states, mask=mask, key_value_states=key_value_states))
