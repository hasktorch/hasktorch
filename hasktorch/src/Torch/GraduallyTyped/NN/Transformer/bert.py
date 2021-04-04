import argparse
import pprint
from typing import Any
import torch
from transformers import AutoTokenizer, BertForMaskedLM

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
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--input", default="The capital of France is [MASK].")
    parser.add_argument("--label", default="The capital of France is Paris.")
    parser.add_argument("--output", default="bert-base-uncased.pt")
    args = parser.parse_args()
    pretty_print(args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenized_inputs = tokenizer([args.input], padding="longest", return_tensors="pt")
    pretty_print(tokenized_inputs)
    back_decoded_inputs = tokenizer.batch_decode(tokenized_inputs['input_ids'], skip_special_tokens=False)
    pretty_print({'back_decoded_inputs': back_decoded_inputs})

    tokenized_labels = tokenizer([args.input], padding="longest", return_tensors="pt")
    pretty_print(tokenized_labels)
    back_decoded_labels = tokenizer.batch_decode(tokenized_labels['input_ids'], skip_special_tokens=False)
    pretty_print({'back_decoded_labels': back_decoded_labels})

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    outputs = model(**tokenized_inputs, labels=tokenized_labels['input_ids'])
    pretty_print({'loss': outputs.loss})

    d = dict(model.state_dict())
    pretty_print({k: v.shape for k, v in d.items()})
    torch.save(d, args.output, _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    main()
