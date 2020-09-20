# Datasets for Examples

A directory for datasets shared across examples. Dataset files in this directory
should not be committed to the repo.

The mnist executable need to have the mnist dataset placed in a local directory ```./data```.

## Run commands

```sh
cd hasktorch/examples
./datasets/download-mnist.sh 
cp -r mnist data
```
