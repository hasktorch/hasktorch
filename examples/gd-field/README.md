In this example, we have used 
- `hasktorch` to compute the gradient field of the following map:
```haskell
f :: Tensor -> Tensor -> Tensor
f x y = F.sin (2 * pit * r) where
    pit = asTensor (pi :: Double)
    r = F.sqrt (x * x + y * y)
```
- `chart` (with `cairo` backend) to plot the vector field.

Please make sure that `cairo` backend is installed in your machine. 
For macos you can have it with 
```
$ brew install pkg-config
$ brew install cairo
```
