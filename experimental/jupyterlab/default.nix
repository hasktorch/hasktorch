{}:
let
  jupyterLib = builtins.fetchGit {
    url = https://github.com/tweag/jupyterWith;
    rev = "70f1dddd6446ab0155a5b0ff659153b397419a2d";
#   See https://github.com/tweag/jupyterWith/pull/77    
#   When you use this on macos, use following branch.
#    url = https://github.com/junjihashimoto/jupyterWith;
#    rev = "6aa22039e278e59d49bff7cd8cc7addb5b01e76f";
  };
  nixpkgsPath = jupyterLib + "/nix";
  haskTorchSrc = builtins.fetchGit {
    url = https://github.com/hasktorch/hasktorch;
    rev = "7e017756fd9861218bf2f804d1f7eaa4d618eb01";
    ref = "master";
  };
  hasktorchOverlay = (import (haskTorchSrc + "/nix/shared.nix") { compiler = "ghc865"; }).overlayShared;
  haskellOverlay = import ./haskell-overlay.nix;
  pkgs = import nixpkgsPath {overlays = [ hasktorchOverlay haskellOverlay ]; config={allowUnfree=true; allowBroken=true;};};

  jupyter = import jupyterLib {pkgs=pkgs;};

  ihaskellWithPackages = jupyter.kernels.iHaskellWith {
      #extraIHaskellFlags = "--debug";
      haskellPackages = pkgs.haskell.packages.ghc865;
      name = "hasktorch";
      packages = p: with p; [
        libtorch-ffi_cpu
        inline-c
        inline-c-cpp
        hasktorch-examples_cpu
        hasktorch_cpu
        matrix
        hmatrix
        monad-bayes
        hvega
        statistics 
        vector
        ihaskell-hvega
        aeson
        aeson-pretty
        formatting
        foldl
        histogram-fill
        funflow
        JuicyPixels
      ];
    };

  jupyterlabWithKernels =
    jupyter.jupyterlabWith {
      kernels = [ ihaskellWithPackages ];
      directory = jupyter.mkDirectoryWith {
        extensions = [
          "jupyterlab-ihaskell"
        ];
      };
    };
in
  jupyterlabWithKernels
