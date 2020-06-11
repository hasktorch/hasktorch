args@{ haskellCompiler ? "ghc883", ...}:

let
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
  hls = (import sources.hls-nix { }).hpkgs.haskell-language-server;
  default = (import ./default.nix { inherit (args); shellBuildInputs = [ niv hls ]; }).defaultCuda102;
in

default.shell

