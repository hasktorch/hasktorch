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
          # tool = compiler-nix-name: name: versionOrArgs:
          #   let
          #     args' = pkgsNew.haskell-nix.haskellLib.versionOrArgsToArgs versionOrArgs;
          #     args = { inherit compiler-nix-name; } // args';
          #   in
          #     (if pkgsNew.haskell-nix.custom-tools ? "${name}"
          #         && pkgsNew.haskell-nix.custom-tools."${name}" ? "${args.version}"
          #       then pkgsNew.haskell-nix.custom-tools."${name}"."${args.version}"
          #       else builtins.trace (pkgsNew.haskell-nix.custom-tools."${name}" ? "${args.version}") pkgsNew.haskell-nix.hackage-tool) (args // { inherit name; });
          # tools = compiler-nix-name: lib.mapAttrs (pkgsNew.haskell-nix.tool compiler-nix-name);
          custom-tools = pkgsOld.haskell-nix.custom-tools // {
            haskell-language-server."0.3.0" = args:
              (pkgsOld.haskell-nix.cabalProject (args // {
                name = "haskell-language-server";
                src = pkgsOld.fetchFromGitHub {
                  owner = "haskell";
                  repo = "haskell-language-server";
                  rev = "d36bb9929fdd0df76f86d3635067400272f68497";
                  sha256 = "0jzj1a15wiwd4wa4wg8x0bpb57g4xrs99yp24623cjcvbarmwjgl";
                  fetchSubmodules = true;
                };
                # lookupSha256 = { location, tag, ... } : {
                #   "https://github.com/wz1000/shake"."fb3859dca2e54d1bbb2c873e68ed225fa179fbef" = "0sa0jiwgyvjsmjwpfcpvzg2p7277aa0dgra1mm6afh2rfnjphz8z";
                #   "https://github.com/peti/cabal-plan"."894b76c0b6bf8f7d2f881431df1f13959a8fce87" = "06iklj51d9kh9bhc42lrayypcpgkjrjvna59w920ln41rskhjr4y";
                #   }."${location}"."${tag}";
                # plan-sha256 = "0a6c4lhnlm2lkic91ips0gb3hqlp3fk2aa01nsa8dhz9l8zg63da";
                modules = [{
                  # Tests don't pass for some reason, but this is a somewhat random revision.
                  packages.haskell-language-server.doCheck = false;
                }];
              })).haskell-language-server.components.exes.haskell-language-server;
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