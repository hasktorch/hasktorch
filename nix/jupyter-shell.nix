let
  extras = (hackage: {
    packages = {
      "ihaskell" = (((hackage.ihaskell)."0.10.1.1").revisions).default;
      "ihaskell-hvega" = (((hackage.ihaskell-hvega)."0.3.1.0").revisions).default;
      "cmdargs" = (((hackage.cmdargs)."0.10.20").revisions).default;
      "ghc-parser" = (((hackage.ghc-parser)."0.2.2.0").revisions).default;
      "haskeline" = (((hackage.haskeline)."0.8.0.0").revisions).default;
      "hlint" = (((hackage.hlint)."3.1.4").revisions).default;
      "haskell-src-exts" = (((hackage.haskell-src-exts)."1.23.1").revisions).default;
      "shelly" = (((hackage.shelly)."1.9.0").revisions).default;
      "strict" = (((hackage.strict)."0.3.2").revisions).default;
      "ipython-kernel" = (((hackage.ipython-kernel)."0.10.2.0").revisions).default;
      "cpphs" = (((hackage.cpphs)."1.20.9.1").revisions).default;
      "uniplate" = (((hackage.uniplate)."1.6.12").revisions).default;
      "extra" = (((hackage.extra)."1.7.3").revisions).default;
      "refact" = (((hackage.refact)."0.3.0.2").revisions).default;
      "filepattern" = (((hackage.filepattern)."0.1.2").revisions).default;
      "ghc-lib-parser-ex" = (((hackage.ghc-lib-parser-ex)."8.10.0.14").revisions).default;
      "ghc-lib-parser" = (((hackage.ghc-lib-parser)."8.10.1.20200523").revisions).default;
      "hscolour" = (((hackage.hscolour)."1.24.4").revisions).default;
      "polyparse" = (((hackage.polyparse)."1.13").revisions).default;
      "alex" = (((hackage.alex)."3.2.5").revisions).default;
      "happy" = (((hackage.happy)."1.19.12").revisions).default;
      "unix-compat" = (((hackage.unix-compat)."0.5.2").revisions).default;
      "lifted-base" = (((hackage.lifted-base)."0.2.3.12").revisions).default;
      "lifted-async" = (((hackage.lifted-async)."0.10.0.6").revisions).default;
      "enclosed-exceptions" = (((hackage.enclosed-exceptions)."1.0.3").revisions).default;
      "cereal-text" = (((hackage.cereal-text)."0.1.0.2").revisions).default;
      "temporary" = (((hackage.temporary)."1.3").revisions).default;
      "uuid" = (((hackage.uuid)."1.3.13").revisions).default;
      "zeromq4-haskell" = (((hackage.zeromq4-haskell)."0.8.0").revisions).default;
      "cryptohash-sha1" = (((hackage.cryptohash-sha1)."0.11.100.1").revisions).default;
      "cryptohash-md5" = (((hackage.cryptohash-md5)."0.11.100.1").revisions).default;
      "network-info" = (((hackage.network-info)."0.2.0.10").revisions).default;
    };
  });
in

# This shell file is specifically to be used for Jupyter Lab.
{ config ? { allowUnfree = true; allowBroken = true; }
# Lets you override niv dependencies of the project without modifications to the source.
, sourcesOverride ? {}
# Enable CUDA support
, cudaSupport ? false
, cudaMajorVersion ? null
, pkgs ? import ./default.nix {
    inherit config sourcesOverride cudaSupport cudaMajorVersion extras;
  }
}:

with pkgs;

let

  ghcWithPackages = packages: (hasktorchHaskellPackages.shellFor {
    buildInputs = [ hasktorchHaskellPackages.ihaskell.components.exes.ihaskell ];
    packages = _: [];
    additional = packages;
  }).ghc;

  iHaskell = jupyterWith.kernels.iHaskellWith {
    # haskellPackages = hasktorchHaskellPackages;
    haskellPackages = { inherit ghcWithPackages; };
    name = "haskell";
    packages = p: with p; [
      hasktorch
      hvega
      ihaskell-hvega
      # matrix
      # hmatrix
      # monad-bayes
      # statistics 
      # vector
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
