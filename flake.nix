{
  description = "Hasktorch";

  inputs = {
    nixpkgs.follows = "haskellNix/nixpkgs-unstable";
    haskell-nix = {
      url = "github:input-output-hk/haskell.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    utils.url = "github:numtide/flake-utils";
    iohkNix = {
      url = "github:input-output-hk/iohk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    libtorch-nix = {
      url = "github:hasktorch/libtorch-nix";
      flake = false;
    };
    jupyterWith = {
      url = "github:tweag/jupyterWith/35eb565c6d00f3c61ef5e74e7e41870cfa3926f7";
      flake = false;
    };
    naersk = {
      url = "github:nmattia/naersk";
      flake = false;
    };
    tokenizers = {
      url = "github:hasktorch/tokenizers";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, haskell-nix, utils, ... }:
    let
      inherit (nxpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      supportedSystems = import ./supported-systems.nix;
      defaultSystem = head supportedSystems;

      overlays = [
        iohkNix.overlays.haskell-nix-extra
        (final: prev: {
          gitref = self.rev or "dirty";
          commonLib = lib
            // iohkNix.lib;
        })
        (import ./nix/pkgs.nix)

      ];
    
    in eachSystem supportedSystems (system:
      let
        pkgs = haskellNix.legacyPackages.${system}.appendOverlays overlays;

        inherit (pkgs.commonLib) eachEnv environments;

        devShell =  import ./shell.nix {
          inherit pkgs cudaSupport cudaMajorVersion;
          withHoogle = true;
        };

        flake = pkgs.hasktorchProject.flake {};

        checks = collectChecks flake.packages;

        exes = collectExes flake.packages;

      in recursiveUpdate flake {
        inherit environments packages checks;

        legacyPackages = pkgs;

        defaultPackage = flake.packages."hasktorch:lib:hasktorch";

        inherit devShell;
      }
    );
}