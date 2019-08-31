{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

with pkgs;

let

  hsenv = haskellPackages.ghcWithPackages (p: with p; [
    cabal-install
    ansi-wl-pprint
    async
    bytestring
    containers
    exceptions
    hashable
    mtl
    optparse-applicative
    parsec
    parsers
    safe-exceptions
    sysinfo
    template-haskell
    transformers
    transformers-compat
    unordered-containers
    vector
  ]);

in

stdenv.mkDerivation {
  name = "hasktorch-dev";
  buildInputs = [ hsenv mkl ];

}