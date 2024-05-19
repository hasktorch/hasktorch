{
  description = "Hasktorch";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = {
    self,
    nixpkgs,
    flake-parts,
    ...
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [
        ./nix/nixpkgs-instances.nix
      ];
      flake = {
        overlays.default = import ./nix/overlay.nix;
      };
      perSystem = {
        pkgsCuda,
        pkgs,
        ...
      }: {
        checks = {
          inherit
            (pkgs.haskell.packages.ghc965)
            codegen
            hasktorch
            hasktorch-gradually-typed
            libtorch-ffi
            libtorch-ffi-helper
            ;
        };
      };
    };
}
