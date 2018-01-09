typeclasses and dimensions TODO:
-----------------------------------

### Modules to hit parity
- remove ./generic/ folder
  + [ ] Storage to subfolder
  + [ ] Merge Tensor folders
  + [ ] remove extra Generic and Generic.Internal files (see Dynamic versions)
- Figure out parity for:
  + [x] Dynamic.Long
  + [ ] Dynamic.Byte
  + [x] Dynamic.Double
  + [ ] Dynamic.DoubleMath
  + [ ] Dynamic.DoubleRandom
  + [ ] Static.Long
  + [ ] Static.Byte
  + [x] Static.DoubleMath
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
  + [x] Blas.hs         -> Needs instances
  + [x] Lapack.hs       -> Needs instances
  + [x] StorageCopy.hs  -> Needs instances
  + [x] Storage.hs      -> This will be the last one for the current PR
  + [x] TensorConv.hs   -> Needs instances
  + [x] TensorCopy.hs   -> Needs instances
  + [x] Tensor.hs       -> Torch.Raw.Tensor        -> THTensor
  + [x] TensorLapack.hs -> Torch.Raw.Tensor.Lapack -> THTensorLapack
  + [x] TensorMath.hs   -> Torch.Raw.Tensor.Math   -> THTensorMath
  + [x] TensorRandom.hs -> Torch.Raw.Tensor.Random -> THTensorRandom
  + [x] Vector.hs       -> Needs instances


