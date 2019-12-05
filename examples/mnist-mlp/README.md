# Untyped MNIST Example

To run, go to the `static-mnist` directory (from the top level `examples/`) and run `setup-data.sh`. (note in the future - will reorganize the location of the mnist data download to avoid confusion).

Build and run this example from inside the `static-mnist` directory (the reason for this is the example loads images from the `data/` relative path which is there with

`stack run mnist-mlp`

The code trains a model then prints some example images to the terminal as ascii art, showing model outputs and ground truth 
