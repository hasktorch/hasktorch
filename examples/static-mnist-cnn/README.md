# Static Mnist MLP

This need to have the mnist dataset placed in a locala directory  ```./data```.

## Run commands

```sh
$ cd hasktorch/examples
$ ./datasets/download-mnist.sh 
$ cp -r mnist data
$ export DEVICE=cpu
$ cabal run static-mnist-cnn
```
