{
  lib,
  config,
  cudaPackages,
  cudaSupport ? config.cudaSupport,
  system,
  haskell,
  python3Packages,
  stdenv,
  tokenizers-haskell,
}: let
  inherit (python3Packages) torch;
  hPkgs = haskell.packages.ghc928;
  hasktorchPackages =
    hPkgs.extend
    (hFinal: hPrev: {
      libtorch-ffi-helper = hFinal.callCabal2nix "libtorch-ffi-helper" ../libtorch-ffi-helper {};
      libtorch-ffi =
        haskell.lib.compose.overrideCabal (drv: {
          # Causes build failure on second compilation.
          enableLibraryProfiling = false;
          configureFlags = [
            "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include"
          ];
        }) (hFinal.callCabal2nix "libtorch-ffi" ../libtorch-ffi {
          inherit torch;
          c10 = torch;
          torch_cpu = torch;
        });
      codegen = hFinal.callCabal2nix "codegen" ../codegen {};
      hasktorch = haskell.lib.compose.overrideCabal (drv: {
        # Expecting data for tests
        doCheck = false;
        enableLibraryProfiling = false;
      }) (hFinal.callCabal2nix "hasktorch" ../hasktorch {});
    });
in
  hasktorchPackages.hasktorch
