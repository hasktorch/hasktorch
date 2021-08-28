{ system ? builtins.currentSystem
, crossSystem ? null
# Lets you customise ghc and profiling (see ./haskell.nix):
, config ? {}
# Lets you override niv dependencies of the project without modifications to the source.
, sourcesOverride ? {}
# Version info, to be passed when not building from a git work tree
, gitrev ? null
# Enable CUDA support
, cudaSupport ? false
, cudaMajorVersion ? null
# Add packages on top of the package set derived from cabal resolution
, extras ? (_: {})
}:

# assert that the correct cuda versions are used
assert cudaSupport -> (cudaMajorVersion == "10" || cudaMajorVersion == "11");

let
  sources = import ./sources.nix { inherit pkgs; }
    // sourcesOverride;
  iohKNix = import sources.iohk-nix {};
  haskellNix = import sources.haskell-nix { inherit system sourcesOverride; };

  # Use haskell.nix default nixpkgs
  nixpkgsSrc = haskellNix.sources.nixpkgs-unstable;

  # for inclusion in pkgs:
  overlays =
    # Haskell.nix (https://github.com/input-output-hk/haskell.nix)
    haskellNix.nixpkgsArgs.overlays
    # override Haskell.nix hackage and stackage sources
    ++ [
      (pkgsNew: pkgsOld: let inherit (pkgsNew) lib; in {
        haskell-nix = pkgsOld.haskell-nix // {
          hackageSrc = sources.hackage-nix;
          stackageSrc = sources.stackage-nix;
        };
      })
    ]
    # the haskell-nix.haskellLib.extra overlay contains some useful extra utility functions for haskell.nix
    ++ iohKNix.overlays.haskell-nix-extra
    # the iohkNix overlay contains nix utilities and niv
    ++ iohKNix.overlays.iohkNix
    # libtorch overlays from libtorch-nix
    # TODO: pull in libGL_driver and cudatoolkit as done in https://github.com/NixOS/nixpkgs/blob/master/pkgs/games/katago/default.nix
    ++ [
      (pkgs: _: with pkgs;
        let libtorchSrc = callPackage "${sources.libtorch-nix}/libtorch/release.nix" { }; in
        if cudaSupport && cudaMajorVersion == "10" then
          let libtorch = libtorchSrc.libtorch_cudatoolkit_10_2; in
          {
            c10 = libtorch;
            torch = libtorch;
            torch_cpu = libtorch;
            torch_cuda = libtorch;
          }
        else if cudaSupport && cudaMajorVersion == "11" then
          let libtorch = libtorchSrc.libtorch_cudatoolkit_11_1; in
          {
            c10 = libtorch;
            torch = libtorch;
            torch_cpu = libtorch;
            torch_cuda = libtorch;
          }
        else
          let libtorch = libtorchSrc.libtorch_cpu; in
          {
            c10 = libtorch;
            torch = libtorch;
            torch_cpu = libtorch;
          }
      )
    ]
    # tokenizers overlays:
    ++ [
      (pkgs: _: with pkgs; {
        naersk = callPackage sources.naersk { };
      })
      (pkgsNew: pkgsOld:
        let tokenizers = (import "${sources.tokenizers}/nix/pkgs.nix") pkgsNew pkgsOld; in
        {
          tokenizers_haskell = tokenizers.tokenizersPackages.tokenizers-haskell;
        }
      )
    ]
    # hasktorch overlays:
    ++ [
      (pkgs: _: with pkgs; {
        inherit gitrev cudaSupport extras;

        # commonLib: mix pkgs.lib with iohk-nix utils and sources:
        commonLib = lib // iohkNix
          // import ./util.nix { inherit haskell-nix; }
          # also expose sources, nixpkgs and overlays
          // { inherit overlays sources nixpkgsSrc; };
      })
      # haskell-nix-ified hasktorch cabal project:
      (import ./pkgs.nix)
    ]
    # jupyterWith overlays:
    ++ [
      (import "${sources.jupyterWith}/nix/python-overlay.nix")
      (import "${sources.jupyterWith}/nix/overlay.nix")
    ];

  pkgs = import nixpkgsSrc {
    inherit system crossSystem overlays;
    config = haskellNix.nixpkgsArgs.config // config;
  };

in pkgs // {
  pre-commit-check =
    let nix-pre-commit-hooks = import (builtins.fetchTarball "https://github.com/cachix/pre-commit-hooks.nix/tarball/master");
    in
    nix-pre-commit-hooks.run {
      src = ./.;
      # If your hooks are intrusive, avoid running on each commit with a default_states like this:
      # default_stages = ["manual" "push"];
      hooks = {
        nixpkgs-fmt.enable = true;
        ormolu.enable = true;
      };
    };
}
