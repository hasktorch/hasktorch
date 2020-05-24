# Distillation Example

Trains MNIST on a larger neural network (MLP 2 hidden layers of 300 dimensions each), distills to a student network (MLP 2 hidden layers of 30 dimensions each) using soft target distribution.

Trains students with soft targets per [Hinton, Vinyals, Dean 2015](https://arxiv.org/abs/1503.02531) with a temperature parameter of 20.

Distillation module operations are decoupled from details of this example. In particular a distillation process is paramaterized as a teacher model, a student model, view functions into both teacher and student (given an input) and a distillation loss that compares the views.

```
data (Parameterized t, Parameterized s) => DistillSpec t s = DistillSpec {
    teacher :: t,
    student :: s,
    teacherView :: t -> Tensor -> ModelView,
    studentView :: s -> Tensor -> ModelView,
    distillLoss :: ModelView -> ModelView -> Tensor
```

Here `ModelView` is just a newtype wrapper around a tensor to communicate intent:

```
newtype ModelView = ModelView { view :: Tensor }
```

A generic offline distillation is then just a function of the following signature:

```
distill
    :: (Parameterized t, Parameterized s, Optimizer o, Dataset d)
    => DistillSpec t s
    -> OptimSpec o
    -> d
    -> IO s
```

Where `OptimSpec` wraps details of the optimizer to be used:

```
data Optimizer o => OptimSpec o = OptimSpec {
    optimizer :: o,
    batchSize :: Int,
    numIters :: Int,
    learningRate :: Tensor
}
```
