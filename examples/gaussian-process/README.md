# Gaussian Process Example

Recommended pre-reading if you are not already familiar with Gaussian 
Processes - [distill article: "A Visual Exploration of Gaussian 
Processes"](https://distill.pub/2019/visual-exploration-gaussian-processes/).

This is a pretty direct implementation of a gaussian process with a radial
basis function along a horizontal axis `t`, with observed points `y` and
unobserved point `x`.

The example conditions on observed data `y` and samples functions at points
`x`. This example uses dynamic tensors, but uses some newtype wrappers to
clarify intent.

# Potential Follow-ups (pull requests welcome)

- A version with typed dimensions (previously implemented for Hasktorch 0.1)
- A scalable variant that doesn't rely on libtorch's cholesky decomposition 
operation but uses SGD methods instead

# Running the Example

Setup environment variables (run this from the top-level hasktorch project 
directory where the `setenv` file is):

```
source setenv
```

Building and running:

```
stack run gaussian-process
```
