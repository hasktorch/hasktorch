let
  pkgs = import <nixpkgs> { };
in
  {
    hasktorch-aten = pkgs.callPackage ./aten.nix { };
  }
