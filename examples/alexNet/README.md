# Untyped Pretrained AlexNet 

This is an implementation of AlexNet model architecture from the ["One weird trick..."](https://arxiv.org/abs/1404.5997) paper using untyped tensors.

Since training of computervision models on ImageNet dataset is a tad bit challanging due to sheer size of dataset.

Hence this model uses pretrained parameters transferred from its pytorch counterpart, which can be extracted using the code in the [gist](https://gist.github.com/SurajK7/90de501ae7cf332b722b06a1dabb527d) or can be downloaded from [here](https://drive.google.com/uc?export=download&confirm=L1Ez&id=1lH-GwVcNY0Jmf10zMRhGbbXaTe-LIH76)

Place the downloaded `alexNet.pt` file in the directory from where the nix-shell is executed and run:

`stack run alexNet`

The code loads the parameters in a skeleton model and runs inference on the tensor `ones [1, 3, 224, 224]`
