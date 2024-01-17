{
  lib,
  config,
  cudaPackages,
  cudaSupport ? config.cudaSupport,
  haskell-nix,
  system,
  torch,
  stdenv,
  tokenizers-haskell,
}:
haskell-nix.cabalProject' {
  src = ./..;
  compiler-nix-name = "ghc924";
  evalSystem = system;
  modules = [
    # Add non-Haskell dependencies
    {
      packages.tokenizers = {
        configureFlags = ["--extra-lib-dirs=${tokenizers-haskell}/lib"];
      };
      packages.libtorch-ffi = {
        configureFlags = [
          "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include"
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
