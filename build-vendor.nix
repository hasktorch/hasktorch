let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      hasktorch-aten = pkgs.callPackage ./vendor/aten.nix
                       { inherit (pkgs.python36Packages) typing pyaml; };
    };
  };
  pkgs = import <nixpkgs> { inherit config; };
in {
  hasktorch-aten = pkgs.hasktorch-aten;
}
