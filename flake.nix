{
  description = "Hasktorch";

  nixConfig = {
    substituters = [
      https://hydra.iohk.io
      https://hasktorch.cachix.org
    ];
    trusted-public-keys = [
      hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ=
      hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc=
    ];
    bash-prompt = "\[\\e[1m\\e[32mdev-hasktorch\\e[0m:\\w\]$ ";
  };

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
      url = "github:hasktorch/libtorch-nix";
      inputs.utils.follows = "haskell-nix/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    jupyterWith = {
      url = "github:tweag/jupyterWith/35eb565c6d00f3c61ef5e74e7e41870cfa3926f7";
      flake = false;
    };
    tokenizers = {
      url = "github:hasktorch/tokenizers/flakes";
      inputs.utils.follows = "haskell-nix/flake-utils";
    };
  };

  outputs = { self, nixpkgs, haskell-nix, libtorch-nix, utils, iohkNix, naersk, tokenizers, ... }: with utils.lib;
    let
      inherit (nixpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      hasktorch.overlays = {
        build-type = final: prev: {
          hasktorch-config = {
            cudaSupport = if prev.lib.hasAttr "cudaSupport" prev.config then prev.config.cudaSupport else false;
            cudaMajorVersion = "11";
            profiling = true;
          };
        };

        dev-tools = final: prev: {
          haskell-nix = prev.haskell-nix // {
            custom-tools = prev.haskell-nix.custom-tools // (prev.callPackage ./nix/haskell-language-server {});
          };
        };

        libtorch-nix = final: prev:
          if !prev.hasktorch-config.cudaSupport then libtorch-nix.overlays.cpu final prev else
          if prev.hasktorch-config.cudaMajorVersion == "10" then libtorch-nix.overlays.cudatoolkit_10_2 final prev else
          libtorch-nix.overlays.cudatoolkit_11_1 final prev;

        hasktorch-project = final: prev: {
          hasktorchProject = import ./nix/haskell.nix ({
            pkgs = prev;
            compiler-nix-name = "ghc901";
            inherit (prev) lib;
            inherit (prev.hasktorch-config) profiling cudaSupport;
          });
        };
      };

    in { inherit (hasktorch) overlays; } // (eachSystem ["x86_64-darwin" "x86_64-linux"] (system:
      let
        overlays = [
          haskell-nix.overlay
          iohkNix.overlays.haskell-nix-extra
          tokenizers.overlay

          hasktorch.overlays.build-type
          hasktorch.overlays.dev-tools
          hasktorch.overlays.libtorch-nix
          hasktorch.overlays.hasktorch-project
        ];

        pkgs = import nixpkgs { inherit system overlays; };

        legacyPkgs = haskell-nix.legacyPackages.${system}.appendOverlays overlays;

        inherit (nixpkgs.lib // iohkNix.lib) eachEnv environments;

        devShell =  pkgs.callPackage ./shell.nix {
          inherit (pkgs.hasktorch-config) cudaSupport cudaMajorVersion;
        };

        flake = pkgs.hasktorchProject.flake {};

        checks = pkgs.haskell-nix.haskellLib.collectChecks' flake.packages;

        exes = pkgs.haskell-nix.haskellLib.extra.collectExes flake.packages;

      in lib.recursiveUpdate flake {
        inherit environments checks legacyPkgs;

        defaultPackage = flake.packages."hasktorch:lib:hasktorch";

        inherit devShell;
      }
    ));
}
