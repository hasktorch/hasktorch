
# This shell file is specifically to be used with Stack.
#
# This file allows using Stack's built-in Nix integration.  This means that you
# can compile hasktorch with Stack by using a command like `stack --nix build`.
# Stack will use Nix to download and build required system libraries (like GHC
# and libtorch), and then build Haskell libraries like normal.
#
# This approach allows for more reproducibility than using Stack without Nix.

{ withHoogle ? false
}@args:

let
  shared = import ../shared.nix {};
  pkgs = shared.defaultCpu.pkgs;
in

with pkgs;

haskell.lib.buildStackProject {
  name = "stack-build-hasktorch";
  ghc = haskell.compiler.ghc8101;
  extraArgs = [
    "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
  ];
  buildInputs = [
    git # needed so that stack can get extra-deps from github
    torch
    zlib
  ];
  inherit withHoogle;
}
