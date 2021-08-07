{ system ? builtins.currentSystem
, crossSystem ? null
, config ? {}
, sourcesOverride ? {}
, gitrev ? null
, cudaSupport ? false
, cudaMajorVersion ? null
, extras ? (_: {})
}:

# assert that the correct cuda versions are used
assert cudaSupport -> (cudaMajorVersion == "10" || cudaMajorVersion == "11");

let
  flakeSources = let
    flakeLock = (builtins.fromJSON (builtins.readFile ../flake.lock)).nodes;
    compat = s: builtins.fetchGit {
      url = "https://github.com/${s.locked.owner}/${s.locked.repo}.git";
      inherit (s.locked) rev;
      ref = s.original.ref or "master";
    };
  in {
    "haskell.nix" = compat flakeLock.haskellNix;
    "iohk-nix" = compat flakeLock.iohkNix;
  };
  sources = flakeSources // sourcesOverride;
  iohkNix = import sources.iohk-nix {};
  haskellNix = import sources.haskell-nix { inherit system sourcesOverride; };

  # Use haskell.nix default nixpkgs
  nixpkgsSrc = haskellNix.sources.nixpkgs-unstable;

  # for inclusion in pkgs:
  overlays =
    # Haskell.nix (https://github.com/input-output-hk/haskell.nix)
    haskellNix.nixpkgsArgs.overlays
    # the haskell-nix.haskellLib.extra overlay contains some useful extra utility functions for haskell.nix
    ++ iohkNix.overlays.haskell-nix-extra
    # the iohkNix overlay contains nix utilities and niv
    ++ iohkNix.overlays.iohkNix
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

in pkgs
