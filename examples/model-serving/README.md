# Model Serving

A series of examples demonstrating integration with servant to serve hasktorch models, serializationto save and load models for deployment, and interop with PyTorch using torchscript and state_dict to save models/load models.

## 1 Simple Computation as an API

*Location*: `examples/model-serving/01-simple-computation`

Example of a embedding a simple computation in Hasktorch. In this example the computatioin just multiplies a value by 2x, but the same serving pattern could be extended to more extensive computations (e.g. a physics simulation).

Run the example: 
```
stack run serve-simple-computation
```


Test with a web browser by entering the url [http://localhost:8081/compute2x/3.1](http://localhost:8081/compute2x/3.1)
which should display `[{"result":[6.2],"msg":"f(x) = 2.0 * x is 6.2 for x = 3.1"}]`

Alternatively, use curl:

```
curl http://localhost:8081/compute2x/3.1
```

which should again display `[{"result":[6.2],"msg":"f(x) = 2.0 * x is 6.2 for x = 3.1"}]`

## 2 Training Followed by Model Inference as an API

*Location*: `examples/model-serving/02-train-inference`

A common pattern in machine learning is to train a model offline and then serve inferences of the model as an API.

We use the same multiple regression example that's under `examples/regression`, but instead of displaying the result at the command line, the model is then made available as an endpoint.

Run the example: 
```
stack run serve-train-inference
```

Test with a web browser [http://localhost:8081/inference/1.0/2.0/3.0](http://localhost:8081/inference/1.0/2.0/3.0).

## 3 Serializing (saving) and Loading a Model

*Location*: `examples/model-serving/03-serialization`

Typically training and serving are run seperately, in which case you want to capture the train model as a serialized artifact.

This example uses `saveParams` and `loadParams` from `Torch.Serialize` to save and load a model within hasktorch.

Run the example: 
```
stack run serve-serialize
```

## 4 PyTorch Interop Part 1: Inference with Torchscript

*Location*: `examples/model-serving/04-python-torchscript`

Run this example from the `04-python-torchscript` directory where the python script is located.

This example demonstrates using Haskell to serve a model trained in PyTorch on the standard MNIST digits classification task..

In this case, where no further training or model introspection is needed from Hasktorch, it's simple to use [torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) to load an opaque computation object to serve a model.

First train the mnist model with pytorch and python 3+:

```
python3 mnist.py
```

The python `mnist.py` script produces two artifacts:

- `mnist.dict.pt` contains just the state_dict of the model.
- `mnist.ts.pt` contains the torchscript graph as well as the state_dict of the model.

Run the example: 
```
stack run python-torchscript
```

## 5 PyTorch Interop Part 2: Importing a Model by Parsing

*Location*: `examples/model-serving/05-python-parse-model`

In some cases, you may want to import a model from PyTorch for further training or development. In this case, rather than loading the model computation as an opaque torchscript object, the model parameters need to be effectively parsed into a corresponding set of Hasktorch parameters (any value implementing the `Parameterized` typeclass).

This example demonstrates a simple version of this using a small regression model.

Run this example from the `05-python-parse-model` directory where the python script is located.

```
python3 simple_model.py
```

The python `simple_model.py` script produces `simple.dict.pt` which contains just the state_dict of a simple linear model. 

Run the example: 
```
stack run python-parse-model
```

The example loads the models into a corresponding hasktorch model by parsing the state_dict object.
