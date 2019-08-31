{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let

  hsenv = haskellPackages.ghcWithPackages (p: with p; [
    cabal-install
    ansi-wl-pprint
    bytestring
    containers
    hashable
    mtl
    parsec
    parsers
    template-haskell
    transformers
    unordered-containers
    vector
  ]);

in

stdenv.mkDerivation {
  name = "hasktorch-dev";
  buildInputs = [ hsenv ];

}