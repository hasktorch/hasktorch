# This shell file is specifically to be used for Jupyter Lab.
{ config ? { allowUnfree = true; allowBroken = true; }
, sourcesOverride ? {}
, cudaSupport ? false
, cudaMajorVersion ? null
, withHoogle ? false
, pkgs ? import ./default.nix {
    inherit config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:
with pkgs;
let
  sources = import ./sources.nix { inherit pkgs; }
    // sourcesOverride;

  jupyter = import sources.jupyterWith {};

  iHaskell = jupyter.kernels.iHaskellWith {
    haskellPackages = hasktorchHaskellPackages;
    name = "haskell";
    packages = p: with p; [
      hasktorch
      # matrix
      # hmatrix
      # monad-bayes
      # hvega
      # statistics 
      # vector
      # ihaskell-hvega
      # aeson
      # aeson-pretty
      # formatting
      # foldl
      # histogram-fill
      # funflow
      # JuicyPixels
    ];
  };

  jupyterEnvironment = jupyter.jupyterlabWith {
    kernels = [ iHaskell ];
    directory = jupyter.mkDirectoryWith {
      extensions = [
        "jupyterlab-ihaskell"
      ];
    };
  };

in

  jupyterEnvironment.env
