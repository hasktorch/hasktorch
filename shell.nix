# This file is used by nix-shell.
# It just takes the shell attribute from default.nix.
{ config ? {}
, sourcesOverride ? {}
# If true, activates CUDA support
, cudaSupport ? false
# If cudaSupport is true, this needs to be set to a valid CUDA major version number, e.g. 10:
# nix-shell --arg cudaSupport true --argstr cudaMajorVersion 10
, cudaMajorVersion ? null
, withHoogle ? false
, pkgs ? import ./nix/default.nix {
    inherit config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:
with pkgs;
let
  # This provides a development environment that can be used with nix-shell or
  # lorri. See https://input-output-hk.github.io/haskell.nix/user-guide/development/
  shell = hasktorchHaskellPackages.shellFor {
    name = "hasktorch-dev-shell";

    # If shellFor local packages selection is wrong,
    # then list all local packages then include source-repository-package that cabal complains about:
    #packages = ps: with ps; [ ];

    tools = {
      cabal = "3.2.0.0";
      ghcide = "0.2.0";
      ormolu = "0.1.2.0";
      brittany = "0.12.1.1";
    };

    # These programs will be available inside the nix-shell.
    buildInputs =
      with haskellPackages; [ hlint weeder ghcid lentil ]
      # TODO: Add additional packages to the shell.
      ++ [ ];

    # Prevents cabal from choosing alternate plans, so that
    # *all* dependencies are provided by Nix.
    # TODO: Set to true as soon as haskell.nix issue #231 is resolved.
    exactDeps = false;

    shellHook = ''
      export CPATH=${torch}/include/torch/csrc/api/include
    '';

    inherit withHoogle;
  };

  # Use to get a shell with niv to update the sources. Launch with
  # nix-shell -A devops ./shell.nix
  devops = stdenv.mkDerivation {
    name = "devops-shell";
    buildInputs = [
      niv
    ];
    shellHook = ''
      echo "DevOps Tools" \
      | ${figlet}/bin/figlet -f banner -c \
      | ${lolcat}/bin/lolcat

      echo "NOTE: you may need to export GITHUB_TOKEN if you hit rate limits with niv"
      echo "Commands:
        * niv update <package> - update package

      "
    '';
  };

in

  shell // { inherit devops; }
