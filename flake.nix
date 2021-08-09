{
  description = "Hasktorch";

  inputs = {
    nixpkgs.follows = "haskell-nix/nixpkgs-unstable";
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

  outputs = { self, nixpkgs, haskell-nix, utils, iohkNix, ... }: with utils.lib;
    let
      inherit (nixpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      #supportedSystems = import ./supported-systems.nix;
      supportedSystems = ["x86_64-linux"];
      defaultSystem = builtins.head supportedSystems;
      gitrev = self.rev or "dirty";

      overlays = [
        haskell-nix.overlay
        iohkNix.overlays.haskell-nix-extra
        (final: prev: {
          inherit gitrev;
          commonLib = lib
            // iohkNix.lib;
        })
        (final: prev: {
          # This overlay adds our project to pkgs
          hasktorchProject = import ./nix/haskell.nix (rec {
            pkgs = prev;
            compiler-nix-name = "ghc8105";
            inherit (prev) lib;
            inherit gitrev;
          });
        })
      ];

    in eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs { inherit system overlays; };

        legacyPkgs = haskell-nix.legacyPackages.${system}.appendOverlays overlays;

        inherit (pkgs.commonLib) eachEnv environments;

        devShell =  import ./shell.nix {
          inherit pkgs;
          withHoogle = true;
        };

        flake = pkgs.hasktorchProject.flake {};

        checks = collectChecks flake.packages;

        exes = collectExes flake.packages;

      in lib.recursiveUpdate flake {
        inherit environments checks legacyPkgs;

        defaultPackage = flake.packages."hasktorch:lib:hasktorch";

        inherit devShell;
      }
    );
}
