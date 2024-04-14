{
  lib,
  config,
  cudaPackages,
  cudaSupport ? config.cudaSupport,
  system,
  haskell,
  python3Packages,
  stdenv,
  pkg-config,
  tokenizers,
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
        doCheck = false;
        enableLibraryProfiling = false;
      }) (hFinal.callCabal2nix "hasktorch" ../hasktorch {});

      indexed-extras =
        hPkgs.indexed-extras.override {
        };
      hasktorch-gradually-typed =
        haskell.lib.compose.overrideCabal (drv: {
          doCheck = false;
          enableLibraryProfiling = false;
        }) (hFinal.callCabal2nix "hasktorch-gradually-typed" ../experimental/gradually-typed {
          type-errors-pretty = haskell.lib.doJailbreak (haskell.lib.dontCheck hPkgs.type-errors-pretty);
        });
    });
in
  hasktorchPackages.hasktorch-gradually-typed
