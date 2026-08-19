# Changelog for libtorch-ffi

## 2.0.2.0

- Add support for Stackage lts-23.24 (GHC 9.8); the previous release used GHC 9.6.
- Expose parameter group interface in `Torch.Internal.Managed.Optim` and `Torch.Internal.Unmanaged.Optim` (supports per-group Adam/AdamW options).
- Implement non-blocking host-to-device transfer.
- Setup: dynamic detection for CUDA package installation, download NVIDIA libs, drop `rpath-link` on macOS.
