# This shell file is specifically to be used with Stack.
#
# This file allows using Stack's built-in Nix integration. This means that you
# can compile hasktorch with Stack by using a command like `stack --nix build`.
# Stack will use Nix to download and build required system libraries (like GHC
# and libtorch), and then build Haskell libraries like normal.
#
# This approach allows for more reproducibility than using Stack without Nix.
{ config ? { }
, sourcesOverride ? { }
, cudaSupport ? false
, cudaMajorVersion ? null
, withHoogle ? false
}:
let
  system = builtins.currentSystem;
  pkgs = (
    import (
      fetchTarball {
        url = "https://github.com/edolstra/flake-compat/archive/99f1c2157fba4bfe6211a321fd0ee43199025dbf.tar.gz";
        sha256 = "0x2jn3vrawwv9xp15674wjz9pixwjyj3j771izayl962zziivbx2";
      }) {
        src =  ./..;
      }
  ).defaultNix.outputs.lib."${system}".cpu.pkgs;
  ghc = pkgs.hasktorchProject.ghcWithPackages (_: []);

  buildInputs = [
    pkgs.git # needed so that stack can get extra-deps from github
    pkgs.torch
    pkgs.zlib
  ];

  stack-shell = pkgs.haskell.lib.buildStackProject {
    inherit ghc;

    name = "hasktorch-stack-dev-shell";

    extraArgs = [
    ];

    inherit buildInputs;

    phases = ["nobuildPhase"];
    nobuildPhase = "echo '${pkgs.lib.concatStringsSep "\n" ([ghc] ++ buildInputs)}' > $out";
    meta.platforms = pkgs.lib.platforms.unix;

    inherit withHoogle;
  };

in

stack-shell
