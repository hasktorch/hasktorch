{
  description = "Hasktorch";

  inputs = {
    nixpkgs.follows = "haskell-nix/nixpkgs-unstable";
    haskell-nix = {
      url = "github:input-output-hk/haskell.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    utils.follows = "haskell-nix/flake-utils";
    iohkNix = {
      url = "github:input-output-hk/iohk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    libtorch-nix = {
      url = "github:stites/libtorch-nix/flakeify";
      inputs.utils.follows = "haskell-nix/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    jupyterWith = {
      url = "github:tweag/jupyterWith/35eb565c6d00f3c61ef5e74e7e41870cfa3926f7";
      flake = false;
    };

    naersk = { # should be moved into a tokenizers flake
      url = "github:nix-community/naersk";
      #flake = false;
    };

    tokenizers = {
      url = "github:hasktorch/tokenizers";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, haskell-nix, libtorch-nix, utils, iohkNix, naersk, tokenizers, ... }: with utils.lib;
    let
      inherit (nixpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      supportedSystems = ["x86_64-darwin" "x86_64-linux"];
      gitrev = self.rev or "dirty";
      cudaSupport = false;
      cudaMajorVersion = "10";

      overlays = [
        haskell-nix.overlay
        iohkNix.overlays.haskell-nix-extra

        (if !cudaSupport then libtorch-nix.overlays.cpu
         else if (cudaMajorVersion == "10") then libtorch-nix.overlays.cudatoolkit_10_2
         else libtorch-nix.overlays.cudatoolkit_11_1)

        (final: prev: {
          inherit gitrev;
          commonLib = lib
            // iohkNix.lib;
        })

        (final: prev: {
          naersk = naersk.lib."${prev.system}";
        })
        (import "${tokenizers}/nix/pkgs.nix")
        (final: prev: {
          tokenizers_haskell = prev.tokenizersPackages.tokenizers-haskell;
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
