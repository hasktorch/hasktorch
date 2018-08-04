# See: https://gist.github.com/CMCDragonkai/41593d6d20a5f7c01b2e67a221aa0330

{ stdenv, cmake, blas, liblapack, gfortran49 }:
let
  pkgs = import <nixpkgs> {};
  python36 = pkgs.python36Packages;
in
  stdenv.mkDerivation rec {
    name = "hasktorch-aten";
    version = "0.0.1";
    src = ./aten;
    buildInputs = [ cmake python36.typing python36.pyaml blas liblapack gfortran49 ];
    builder = ./builder.sh;
    inherit cmake;
  }


