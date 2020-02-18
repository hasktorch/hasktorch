In this example, we have used 
- `hasktorch` to compute the gradient field of the following map:
```haskell
f :: Tensor -> Tensor -> Tensor
f x y = F.sin (2 * pit * r) where
    pit = asTensor (pi :: Double)
    r = F.sqrt (x * x + y * y)
```
- `hvega` to plot the heatmap corresponding to the norm of the
  gradient field.

The following command produces an html file including the plot:
```
$ stack Main.hs
```
