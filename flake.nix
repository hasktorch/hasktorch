{
  description = "Hasktorch";

  inputs = {
    tokenizers = {
      url = "github:collinarnett/tokenizers/collinarnett/fix/pkgconfig";
    };
    nixpkgs.follows = "tokenizers/nixpkgs";
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
        checks = {
          inherit
            (pkgs.haskell.packages.ghc928)
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
