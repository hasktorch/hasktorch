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

tokenizer = AutoTokenizer.from_pretrained('t5-small')

def print_example(model):
    prefix = "translate English to German: "
    # prefix = "summarize: "

    # inputs = [
    #     prefix + "Studies have shown that owning a dog is good for you.",
    #     prefix + "Studies have shown that owning a dog is good for you and your dog.",
    #     prefix + "You're full of shit!"]
    # tokenized_inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    # decoder_inputs = [
    #     "Studien haben gezeigt, dass das Besitzen eines Hundes gut für Sie ist.",
    #     "Studien haben gezeigt, dass das Besitzen eines Hundes gut für Sie und Ihren Hund ist.",
    #     "Du bist voller Scheiße!"]
    # tokenized_decoder_inputs = tokenizer(decoder_inputs, padding="longest", return_tensors="pt")
    # outputs = model(
    #     input_ids=tokenized_inputs.input_ids,
    #     attention_mask=tokenized_inputs.attention_mask,
    #     decoder_input_ids=tokenized_decoder_inputs.input_ids,
    #     decoder_attention_mask=tokenized_decoder_inputs.attention_mask)
    # print(tokenized_inputs)
    # print(tokenized_decoder_inputs)
    # print(outputs.logits)

    inputs = [
        prefix + "Studies have shown that owning a dog is good for you",
        prefix + "You're full of shit!",
        prefix + "That sounds pretty great, thanks for sharing! As Austin said, "
               + "we are looking into bringing some of the huggingface model "
               + "architectures (T5, BERT, GPT, etc.) and functionality "
               + "(tokenization, training, fine tuning, etc.) to hasktorch. "
               + "I’m currently also building a new API for hasktorch that "
               + "interpolates between the already existing untyped and the typed API. "
               + "I've reimplemented the T5 model architecture in that new API, "
               + "and I'm able to use Google's model checkpoints to decode text "
               + "(translations, summaries, etc.) from it. Let me know if you want "
               + "to discuss any of that.",
        prefix + "'"
               + "select t2.name, count(*) from concert as t1 "
               + "join stadium as t2 on t1.stadium_id = t2.stadium_id "
               + "group by t1.stadium_id"
               + "'",
        prefix + "\" "
               + "SELECT T2.name, COUNT(*) FROM concert AS T1 "
               + "JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id "
               + "GROUP BY t1.stadium_id "
               + "\""]
    tokenized_inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    print(tokenized_inputs.input_ids)
    print(tokenized_inputs.input_ids.size())
    print(list(map(tokenizer.decode, tokenized_inputs.input_ids)))
    output = model.generate(
        input_ids=tokenized_inputs.input_ids,
        attention_mask=tokenized_inputs.attention_mask,
        num_beams=1,
        max_length=512,
        do_sample=False,
        repetition_penalty=1,
        no_repeat_ngram_size=0)
    print(output)
    decoded = list(map(tokenizer.decode, output))
    print(decoded)


def load_print_and_save(model_string):
    # model = T5Model.from_pretrained('t5-small', torchscript=False)
    model = T5ForConditionalGeneration.from_pretrained(model_string, torchscript=False)
    model.eval()

    print_example(model)

    d = dict(model.state_dict())
    # for k, v in d.items():
    #     print("{}: {}".format(k, v.shape))
    torch.save(d, model_string + ".pt", _use_new_zipfile_serialization=True)

for model_string in ['t5-small']: #, 't5-base', 't5-large', 't5-3b', 't5-11b']:
    load_print_and_save(model_string)

# for i in range(40000):
#     print("({}, \"{}\"),".format(i, tokenizer.convert_ids_to_tokens(i)))
