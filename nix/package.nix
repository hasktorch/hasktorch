{cudaSupport}: final: prev: {
  libtorch-bin = prev.libtorch-bin.overrideAttrs (old: {
    installPhase =
      old.installPhase
      + ''

        pushd $dev/include/torch
        for i in csrc/api/include/torch/* ; do
          ln -s $i
        done
        popd
      '';
  });
  libtorch = final.libtorch-bin;
  torch = final.libtorch-bin;
  c10 = final.libtorch-bin;
  torch_cpu = final.libtorch-bin;
  torch_cuda = final.libtorch-bin;
  hasktorch = final.haskell-nix.cabalProject' {
    src = ./..;
    compiler-nix-name = "ghc924";

    modules = [
      # Enable profiling
      {
        enableLibraryProfiling = true;
      }

      # Add non-Haskell dependencies
      {
        packages.tokenizers = {
          configureFlags = [
            "--extra-lib-dirs=${final.tokenizers-haskell}/lib"
          ];
        };
        packages.libtorch-ffi = {
          configureFlags = with final; [
            "--extra-lib-dirs=${torch.out}/lib"
            "--extra-lib-dirs=${cudaPackages.cuda_nvrtc}/lib"
            "--extra-include-dirs=${torch.dev}/include"
            "--extra-include-dirs=${cudaPackages.cuda_nvrtc}/include"
          ];
          flags = {
            cuda = cudaSupport;
            rocm = false;
            gcc = !cudaSupport && prev.stdenv.hostPlatform.isDarwin;
          };
        };
      }
    ];
  };
}
