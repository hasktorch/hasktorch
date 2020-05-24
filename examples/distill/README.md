# Distillation Example

Trains MNIST on a larger neural network (MLP 2 hidden layers of 300 dimensions each), distills to a student network (MLP 2 hidden layers of 30 dimensions each) using soft target distribution.

Trains students with soft targets per [Hinton, Vinyals, Dean 2015](https://arxiv.org/abs/1503.02531) with a temperature parameter of 20.

Distillation module operations are decoupled from details of this example. In particular a distillation process is paramaterized as a teacher model, a student model, view functions into both teacher and student (given an input) and a distillation loss that compares the views.

```
data (Parameterized t, Parameterized s) => DistillSpec t s = DistillSpec {
    teacher :: t,
    student :: s,
    teacherView :: t -> Tensor -> Tensor,
    studentView :: s -> Tensor -> Tensor,
    distillLoss :: Tensor -> Tensor -> Tensor
}
```

A generic offline distillation is then a function of the following signature:

```
distill
    :: (Parameterized t, Parameterized s, Optimizer o, Dataset d)
    => DistillSpec t s
    -> OptimSpec o
    -> d
    -> IO s
```


