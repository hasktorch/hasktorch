args@{ haskellCompiler ? "ghc883", ...}:

let
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
  hls = (import sources.hls-nix { }).hpkgs.haskell-language-server;
  defaultHsPkgs = (import ./default.nix args).hsPkgsCuda102;
in

defaultHsPkgs.shellFor {
  withHoogle = true;
  tools = { cabal = "3.2.0.0"; };
  buildInputs = [
    niv
    hls
  ];
  exactDeps = true;
}

