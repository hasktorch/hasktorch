# This shell file is specifically to be used for Jupyter Lab.
{ config ? { allowUnfree = true; allowBroken = true; }
, sourcesOverride ? {}
, cudaSupport ? false
, cudaMajorVersion ? null
, pkgs ? import ./default.nix {
    inherit config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:
with pkgs;
let

  # hsPkgs = {
  #   ghcWithPackages = packages: (hasktorchHaskellPackages.shellFor {
  #     # buildInputs =
  #     #   with haskellPackages; [ ihaskell ];
  #     packages = _: [];
  #     additional = p: packages (p // (with haskellPackages; { inherit ihaskell; }));
  #   }).ghc;
  # };

  iHaskell = jupyterWith.kernels.iHaskellWith {
    # haskellPackages = hasktorchHaskellPackages;
    # haskellPackages = hsPkgs;
    inherit haskellPackages;
    name = "haskell";
    packages = p: with p; [
      # hasktorch
      # matrix
      # hmatrix
      # monad-bayes
      hvega
      # statistics 
      # vector
      ihaskell-hvega
      # aeson
      # aeson-pretty
      # formatting
      # foldl
      # histogram-fill
      # funflow
      # JuicyPixels
    ];
  };

  jupyterEnvironment = jupyterWith.jupyterlabWith {
    kernels = [ iHaskell ];
    directory = jupyterWith.mkDirectoryWith {
      extensions = [
        "jupyterlab-ihaskell"
      ];
    };
  };

in

  jupyterEnvironment.env
