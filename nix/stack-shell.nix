
# This shell file is specifically to be used with Stack.
#
# This file allows using Stack's built-in Nix integration. This means that you
# can compile hasktorch with Stack by using a command like `stack --nix build`.
# Stack will use Nix to download and build required system libraries (like GHC
# and libtorch), and then build Haskell libraries like normal.
#
# This approach allows for more reproducibility than using Stack without Nix.
{ config ? {}
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
  ghc = hasktorchHaskellPackages.ghcWithPackages (_: []);

  buildInputs = [
    git # needed so that stack can get extra-deps from github
    torch
    zlib
  ];

  stack-shell = haskell.lib.buildStackProject {
    inherit ghc;

    name = "hasktorch-stack-dev-shell";
    
    extraArgs = [
      "--extra-include-dirs=${torch}/include/torch/csrc/api/include"
    ];
    
    inherit buildInputs;

    phases = ["nobuildPhase"];
    nobuildPhase = "echo '${lib.concatStringsSep "\n" ([ghc] ++ buildInputs)}' > $out";
    meta.platforms = lib.platforms.unix;
    
    inherit withHoogle;
  };

in

  stack-shell