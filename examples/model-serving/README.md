# Model Serving

A series of examples demonstrating integration with servant to serve hasktorch models as API.

## 01 Simple Computation

Example of a embedding a simple computation in Hasktorch. In this example the computatioin just multiplies a value by 2x, but the same serving pattern could be extended to more extensive computations (e.g. a physics simulation).

location: `examples/model-serving/01-simple-computation`

run using: `stack run serve-simple-computation`

Test with a web browser by entering the url [http://localhost:8081/compute2x/3.1](http://localhost:8081/compute2x/3.1)
which should display `[{"result":[6.2],"msg":"f(x) = 2.0 * x is 6.2 for x = 3.1"}]`

Alternatively, use curl:

```
curl http://localhost:8081/compute2x/3.1
```

which should again display `[{"result":[6.2],"msg":"f(x) = 2.0 * x is 6.2 for x = 3.1"}]`

## 2 Training and Inference

A common pattern in machine learning is to train a model offline and then serve inferences of the model as an API.

We use the same multiple regression example that's under `examples/regression`, but instead of displaying the result at the command line, the model is then made available as an endpoint.

location: `examples/model-serving/02-train-inference`

run using: `stack run serve-train-inference`

Test with a web browser [http://localhost:8081/inference/1.0/2.0/3.0](http://localhost:8081/inference/1.0/2.0/3.0).

## 3 Serialization

WIP TODO

## 4 Torchscript

WIP TODO

## 5 PyTorch -> Hasktorch

WIP TODO

## 6 A Small Web Application

WIP TODO
