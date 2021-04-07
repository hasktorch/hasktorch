import argparse
import pprint
from typing import Any
import torch
from transformers import AutoTokenizer, PegasusForConditionalGeneration


def pretty_convert(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    else:
        return x


def pretty_print(x: dict) -> None:
    y = {k: pretty_convert(v) for k, v in x.items()}
    pp = pprint.PrettyPrinter(indent=4, compact=True, width=120)
    pp.pprint(y)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/pegasus-xsum")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="pegasus-xsum.pt")
    args = parser.parse_args()
    pretty_print(args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenized_inputs = tokenizer([args.input], padding="longest", return_tensors="pt")
    pretty_print(tokenized_inputs)
    back_decoded = tokenizer.batch_decode(
        tokenized_inputs["input_ids"], skip_special_tokens=False
    )
    pretty_print({"back_decoded": back_decoded})

    model = PegasusForConditionalGeneration.from_pretrained(
        args.model, torchscript=False
    )
    model.eval()

    # generated_ids = model.generate(
    #     input_ids=tokenized_inputs.input_ids,
    #     attention_mask=tokenized_inputs.attention_mask,
    #     num_beams=1,
    #     max_length=512,
    #     do_sample=False,
    #     repetition_penalty=1,
    #     no_repeat_ngram_size=0,
    #     length_penalty=1,
    # )

    # decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    # pretty_print({"generated_ids": generated_ids, "decoded": decoded})

    # generated_ids = [   0, 139, 37695, 6817, 108, 836, 115, 39290, 108, 117, 156, 113, 109, 205, 1808, 17501, 115, 3165, 107, 1]
    decoded = '<pad> The Eiffel Tower, built in 1889, is one of the most famous landmarks in Paris.</s>'

    tokenized_decoder_inputs = tokenizer(
        decoded, padding="longest", return_tensors="pt"
    )
    print(tokenized_decoder_inputs)

    outputs = model(
        input_ids=tokenized_inputs.input_ids,
        attention_mask=tokenized_inputs.attention_mask,
        decoder_input_ids=tokenized_decoder_inputs.input_ids,
        decoder_attention_mask=tokenized_decoder_inputs.attention_mask,
        labels=None,
        return_dict=True,
    )
    print(outputs.logits)

    d = dict(model.state_dict())
    pretty_print({k: v.shape for k, v in d.items()})
    torch.save(d, args.output, _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    main()
