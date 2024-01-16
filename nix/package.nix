{
  lib,
  config,
  cudaPackages,
  cudaSupport ? config.cudaSupport,
  haskell-nix,
  python3Packages,
  stdenv,
  tokenizers-haskell,
}:
let
  inherit (python3Packages) torch;
in

haskell-nix.cabalProject' {
  src = ./..;
  compiler-nix-name = "ghc924";

  modules = [
    # Enable profiling
    { enableLibraryProfiling = true; }

    # Add non-Haskell dependencies
    {
      packages.tokenizers = {
        configureFlags = [ "--extra-lib-dirs=${tokenizers-haskell}/lib" ];
      };
      packages.libtorch-ffi = {
        configureFlags = [
          "--extra-lib-dirs=${lib.getLib torch}/lib"
          "--extra-lib-dirs=${cudaPackages.cuda_nvrtc}/lib"
          "--extra-include-dirs=${lib.getDev torch}/include"
          "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include"
          "--extra-include-dirs=${cudaPackages.cuda_nvrtc}/include"
        ];
        flags = {
          cuda = cudaSupport;
          rocm = false;
          gcc = !cudaSupport && stdenv.hostPlatform.isDarwin;
        };
      };
    }
  ];
}
