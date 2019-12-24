
import altair as alt
from numpy import array
import pandas as pd
import torch
from torch import nn


def extract_arrays(layer: nn.Module) -> dict:
    """ given a layer return a dict :: param_name -> array of values"""
    return {
        name: values.flatten().detach().numpy()
        for (name, values) in layer.named_parameters()
    }


def to_dataframe(layer_name: str, arr_dict: dict) -> pd.DataFrame:
    """ given a dict :: param_name -> array of values, construct a dataframe"""
    lst = []
    for k, v in arr_dict.items():
        lst.append(
            pd.DataFrame(
                {
                    "layer": layer_name,
                    "param": k,
                    "layer_param": layer_name + "_" + k,
                    "values": v,
                }
            )
        )
    return pd.concat(lst)


def make_histogram(values_df: pd.DataFrame, sub_sample=100):
    # not quite working as intended yet
    grouped = values_df.groupby("layer_param")
    values_subsample = grouped.apply(lambda x: x.sample(sub_sample, replace=True))
    return (
        alt.Chart(values_subsample.sample(5000))
        .mark_bar()
        .encode(alt.X("values", bin=alt.Bin(step=0.01)), y="count()")
        .facet(facet="layer_param", columns=8)
    )


if __name__ == "__main__":

    test_layers = {
        "linear_50_100": nn.Linear(50, 100),
        "linear_100_100": nn.Linear(100, 100),
        "linear_100_100": nn.Linear(200, 100),
        "linear_400_100": nn.Linear(400, 100),
        # "conv1d": nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5),
        # "conv2d": nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5),
        # "conv3d": nn.Conv3d(in_channels=30, out_channels=10, kernel_size=5),
        # "lstm": nn.LSTM(input_size=50, hidden_size=30, num_layers=2),
        # "gru": nn.GRU(input_size=50, hidden_size=30, num_layers=2),
        # "transformer": nn.Transformer(),
    }

    values = {k: extract_arrays(v) for (k, v) in test_layers.items()}
    values_df = pd.concat(
        [to_dataframe(k, extract_arrays(v)) for (k, v) in test_layers.items()]
    )

    print(values_df)

    # plt = make_histogram(values_df)
    # plt.serve()
