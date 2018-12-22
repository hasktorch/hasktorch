{ cudaSupport ? false, mklSupport ? true }:

let
  release = import ./build.nix { inherit cudaSupport mklSupport; };
  pkgs = import <nixpkgs> {};
  lib = pkgs.lib;
in

with release;

pkgs.mkShell {
  name = "hasktorch-dev-environment";
  LD_LIBRARY_PATH    = "${hasktorch-aten}/lib";
  C_INCLUDE_PATH     = "${hasktorch-aten}/include";
  CPLUS_INCLUDE_PATH = "${hasktorch-aten}/include";
  buildInputs = [
    hasktorch-aten
    hasktorch-codegen
    hasktorch-types-th
    hasktorch-ffi-th
    ] ; # ++ (lib.optionals cudaSupport [ hasktorch-types-thc hasktorch-ffi-thc ]);
}

