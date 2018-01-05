typeclasses and dimensions TODO:
-----------------------------------

### Modules to hit parity
- remove ./generic/ folder
  + [ ] Storage to subfolder
  + [ ] Merge Tensor folders
  + [ ] remove extra Generic and Generic.Internal files (see Dynamic versions)
- Figure out parity for:
  + [ ] Dynamic.Long
  + [ ] Dynamic.Byte
  + [ ] Dynamic.DoubleMath
  + [ ] Dynamic.DoubleRandom
  + [ ] Static.Long
  + [ ] Static.Byte
  + [ ] Static.DoubleMath
  + [ ] Static.DoubleRandom
  + [ ] Static.DoubleLapack
- [ ] Come up with CORE typeclasses (not Raw Typeclasses like the current situation).

### Port from raw (FILE THIS INTO ISSUES)
- Top-level files (probably generic functions):
  + [ ] THDiskFile.hs
  + [ ] THFile.hs
  + [ ] THLogAdd.hs
  + [ ] THMemoryFile.hs
  + [ ] THRandom.hs
  + [ ] THSize.hs
  + [ ] THStorage.hs
- Typeclasses:
  + [ ] Blas.hs
  + [ ] Lapack.hs
  + [ ] StorageCopy.hs
  + [ ] Storage.hs      -> This will be the last one for the current PR
  + [ ] TensorConv.hs
  + [ ] TensorCopy.hs
  + [x] Tensor.hs       -> Torch.Core.Tensor.Generic.Ops    (should be in Torch.Raw)
  + [x] TensorLapack.hs -> Torch.Core.Tensor.Generic.Lapack (should be in Torch.Raw)
  + [x] TensorMath.hs   -> Torch.Core.Tensor.Generic.Math   (should be in Torch.Raw)
  + [x] TensorRandom.hs -> Torch.Core.Tensor.Generic.Random (should be in Torch.Raw)
  + [ ] Vector.hs


