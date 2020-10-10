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
assert cudaSupport -> (cudaMajorVersion == "9" || cudaMajorVersion == "10");

let
  sources = import ./sources.nix { inherit pkgs; }
    // sourcesOverride;
  iohKNix = import sources.iohk-nix {};
  haskellNix = (import sources."haskell.nix" { inherit system sourcesOverride; }).nixpkgsArgs;
  # use our own nixpkgs if it exist in our sources,
  # otherwise use iohkNix default nixpkgs.
  nixpkgs = sources.nixpkgs-2003 or
    (builtins.trace "Using IOHK default nixpkgs" iohKNix.nixpkgs);

  # for inclusion in pkgs:
  overlays =
    # Haskell.nix (https://github.com/input-output-hk/haskell.nix)
    haskellNix.overlays
    # override Haskell.nix hackage and stackage sources
    ++ [
      (pkgsNew: pkgsOld: let inherit (pkgsNew) lib; in {
        haskell-nix = pkgsOld.haskell-nix // {
          hackageSrc = sources.hackage-nix;
          stackageSrc = sources.stackage-nix;
          custom-tools = pkgsOld.haskell-nix.custom-tools // {
            haskell-language-server."0.5.0" = args:
              let
                project = pkgsOld.haskell-nix.project' (args // {
                  src = pkgsOld.evalPackages.fetchgit {
                    url = "https://github.com/haskell/haskell-language-server.git";
                    fetchSubmodules = true;
                    rev = "14497f2503a2a0d389fabf3b146d674b9af41a34";
                    sha256 = "0vkh5ff6l5wr4450xmbki3cfhlwf041fjaalnwmj7zskd72s9p7p";
                  };
                  projectFileName = "cabal.project";
                  cabalProjectLocal = ''
                    allow-newer: diagrams-svg:base, monoid-extras:base, svg-builder:base,
                      diagrams-lib:base, dual-tree:base, active:base, diagrams-core:base,
                      diagrams-contrib:base, force-layout:base, diagrams-postscript:base,
                      statestack:base

                    package haskell-language-server
                      flags: -agpl
                  '';
                });
              in
                project.hsPkgs.haskell-language-server.components.exes.haskell-language-server;
          };
        };
      })
    ]
    # the haskell-nix.haskellLib.extra overlay contains some useful extra utility functions for haskell.nix
    ++ iohKNix.overlays.haskell-nix-extra
    # the iohkNix overlay contains nix utilities and niv
    ++ iohKNix.overlays.iohkNix
    # libtorch overlays from pytorch-world
    # TODO: pull in libGL_driver and cudatoolkit as done in https://github.com/NixOS/nixpkgs/blob/master/pkgs/games/katago/default.nix
    ++ [
      (pkgs: _: with pkgs;
        let libtorchSrc = callPackage "${sources.pytorch-world}/libtorch/release.nix" { }; in
        if cudaSupport && cudaMajorVersion == "9" then
          let libtorch = libtorchSrc.libtorch_cudatoolkit_9_2; in
          {
            c10 = libtorch;
            torch = libtorch;
            torch_cpu = libtorch;
            torch_cuda = libtorch;
          }
        else if cudaSupport && cudaMajorVersion == "10" then
          let libtorch = libtorchSrc.libtorch_cudatoolkit_10_2; in
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
    # hasktorch overlays:
    ++ [
      (pkgs: _: with pkgs; {
        inherit gitrev cudaSupport extras;

        # commonLib: mix pkgs.lib with iohk-nix utils and sources:
        commonLib = lib // iohkNix
          // import ./util.nix { inherit haskell-nix; }
          # also expose sources, nixpkgs and overlays
          // { inherit overlays sources nixpkgs; };
      })
      # haskell-nix-ified hasktorch cabal project:
      (import ./pkgs.nix)
    ]
    # jupyterWith overlays:
    ++ [
      (import "${sources.jupyterWith}/nix/python-overlay.nix")
      (import "${sources.jupyterWith}/nix/overlay.nix")
    ];

  pkgs = import nixpkgs {
    inherit system crossSystem overlays;
    config = haskellNix.config // config;
  };

in pkgs
