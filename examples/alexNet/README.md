# Untyped Pretrained AlexNet 

This is an implementation of AlexNet model architecture from the ["One weird trick..."](https://arxiv.org/abs/1404.5997) paper using untyped tensors.

Since training of computer vision models on ImageNet dataset is a tad bit challanging due to the sheer size of dataset.

Hence this model uses pretrained parameters transferred from its pytorch counterpart, which can be extracted using the code in the [gist](https://gist.github.com/SurajK7/90de501ae7cf332b722b06a1dabb527d) or can be downloaded from [here](https://drive.google.com/uc?export=download&confirm=L1Ez&id=1lH-GwVcNY0Jmf10zMRhGbbXaTe-LIH76).

Before running this example place the downloaded `alexNet.pt` file in the alexNet directory.

From the nix-shell present in examples directory run:

`cabal v1-build`

`./dist/build/alexNet/alexNet`

The code loads the parameters in a skeleton model and runs inference on a 224 * 224 pixels image.To run inference on an image of your choice place the image in alexNet directory and enter its name alongwith image file extension when prompted during execution. 

For now images of dimension 224 * 224 and of format PNG, bitmap and JPEG are supported.