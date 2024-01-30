{
  description = "Hasktorch";

  inputs = {
    tokenizers = {
      url = "github:hasktorch/tokenizers/flakes";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };
  nixConfig.extra-substituters = [
    "https://cuda-maintainers.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
  ];

  outputs = {
    self,
    tokenizers,
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
        system,
        ...
      }: {
        # Working
        packages.default = pkgs.callPackage ./nix/package.nix {};
        # Complains about constraints-deriving
        packages.broken = pkgs.haskell.packages.ghc928.hasktorch;
      };
    };
}
