{
  description = "Hasktorch";

  inputs = {
    tokenizers = {
      url = "github:hasktorch/tokenizers/flakes";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    haskell-nix.url = "github:input-output-hk/haskell.nix";
    nixpkgs.follows = "haskell-nix/nixpkgs-2305";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    nixgl = {
      url = "github:nix-community/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig.extra-substituters = [
    "https://cuda-maintainers.cachix.org"
    "https://cache.iog.io"
  ];
  nixConfig.extra-trusted-public-keys = [
    "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    "hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ="
  ];

  outputs = {
    self,
    tokenizers,
    haskell-nix,
    nixpkgs,
    flake-parts,
    nixgl,
    ...
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [
        ./nix/nixpkgs-instances.nix
      ];
      perSystem = {
        pkgsCuda,
        pkgs,
        system,
        ...
      }: let
        hasktorchFor = p:
          pkgs.hasktorch.shellFor {
            name = "hasktorch-dev-shell";
            exactDeps = true;
            withHoogle = true;

            tools.cabal = "latest";
            tools.haskell-language-server = "latest";
          };
      in {
        packages.cuda = pkgsCuda.hasktorch;
        devShells.default = hasktorchFor pkgs;
        devShells.cuda = pkgsCuda.hasktorch.shellFor {
          name = "hasktoch-dev-shell";
          exactDeps = true;
          withHoogle = true;

          tools.cabal = "latest";
          tools.haskell-language-server = "latest";
          #     shellHook = ''
          #       export LD_LIBRARY_PATH=$(${pkgsCuda.nixgl.auto.nixGLDefault}/bin/nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
          #     '';
        };
      };
    };
}
