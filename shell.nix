{ cudaSupport ? false, mklSupport ? true, releasePkgs ? true }:

let
  release = import ./build.nix { inherit cudaSupport mklSupport; };
  pkgs = if releasePkgs then release.pkgs else (import <nixpkgs> {});
  lib = pkgs.lib;
in

with release;

pkgs.mkShell {
  name = "hasktorch-dev-environment";
  PATH               = "${cudatoolkit}/bin";
  LD_LIBRARY_PATH    = "${hasktorch-aten}/lib:${cudatoolkit}/lib64";
  C_INCLUDE_PATH     = "${hasktorch-aten}/include:${cudatoolkit}/include";
  CPLUS_INCLUDE_PATH = "${hasktorch-aten}/include:${cudatoolkit}/include";
  buildInputs = [
    pkgs.cabal-install
    (pkgs.haskellPackages.ghcWithPackages (self: with self; [ hspec-discover ]))

    hasktorch-aten
    hasktorch-codegen
    hasktorch-types-th
    hasktorch-ffi-th
    # these seem to have a runtime dependency error
    ] ; # ++ (lib.optionals cudaSupport [ hasktorch-types-thc hasktorch-ffi-thc ]);
}

