# Changelog for hasktorch

## 0.2.2.0

- Fix `calculateFan`: the receptive field size wrongly included the input channel dimension, shrinking kaiming/xavier init and the default conv layer init by a factor of `sqrt in_channels` for tensors with more than 2 dimensions.
- Add support for Stackage lts-23.24 (GHC 9.8); the previous release used GHC 9.6.
- Add `BFloat16` to `DTypeIsFloatingPoint` and `KnownDType`.
- Implement the AdamW optimizer.
- Expose the parameter group interface for `CppOptim` (per-group Adam/AdamW options).
- Implement non-blocking host-to-device transfer.
- Remove partial signatures from `Torch.Typed.NN.Convolution`.
