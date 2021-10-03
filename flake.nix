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

      build-config.dev     = import ./nix/dev-config.nix; # only used to generate devshell
      build-config.cpu     = { profiling = true; cudaSupport = false; cudaMajorVersion = "invalid"; };
      build-config.cuda-10 = { profiling = true; cudaSupport = true;  cudaMajorVersion = "10"; };
      build-config.cuda-11 = { profiling = true; cudaSupport = true;  cudaMajorVersion = "11"; };

      hasktorch.overlays = {
        cpu-config     = final: prev: { hasktorch-config = build-config.cpu; }     // (libtorch-nix.overlays.cpu final prev);
        cuda-10-config = final: prev: { hasktorch-config = build-config.cuda-10; } // (libtorch-nix.overlays.cudatoolkit_10_2 final prev);
        cuda-11-config = final: prev: { hasktorch-config = build-config.cuda-11; } // (libtorch-nix.overlays.cudatoolkit_11_1 final prev);

        dev-tools = final: prev: {
          haskell-nix = prev.haskell-nix // {
            custom-tools = prev.haskell-nix.custom-tools // (prev.callPackage ./nix/haskell-language-server {});
          };
        };

        hasktorch-project = final: prev: {
          hasktorchProject = import ./nix/haskell.nix ({
            pkgs = prev;
            compiler-nix-name = "ghc901";
            inherit (prev) lib;
            inherit (prev.hasktorch-config) profiling cudaSupport;
          });
        };
      };

      generic-pkgset = system: ty:
        let
          overlays = [
            haskell-nix.overlay
            iohkNix.overlays.haskell-nix-extra
            tokenizers.overlay

            hasktorch.overlays."${ty}-config"
            hasktorch.overlays.dev-tools
            hasktorch.overlays.hasktorch-project
          ];
        in
          { pkgs = import nixpkgs { inherit system overlays; };
            legacyPkgs = haskell-nix.legacyPackages.${system}.appendOverlays overlays;
          };

    in { inherit (hasktorch) overlays; } // (eachSystem [ "x86_64-linux" "x86_64-darwin" ] (system:
      let
        mk-pkgset = generic-pkgset system;

        pkgset = rec {
          cpu     = mk-pkgset "cpu";
          cuda-10 = mk-pkgset "cuda-10";
          cuda-11 = mk-pkgset "cuda-11";
        };

        mapper = name-pred: name-map: with lib.attrsets; mapAttrs' (name: nameValuePair (if (name-pred name) then (name-map name) else name));
        mapper2 = name-pred: name-map: with lib.attrsets;
          mapAttrs' (p: v: nameValuePair p (mapper name-pred name-map v));

        build-flake = ty: with lib.strings; with lib.lists;
          let
            inherit (pkgset.${ty}) pkgs;
            pkg-flake = pkgs.hasktorchProject.flake {};
          in
            mapper2
                      #(n: head (splitString ":" n) == "hasktorch")
                      (n: hasInfix ":" n && head (splitString ":" n) != "codegen")
                      (n: let parts = splitString ":" n; in concatStringsSep ":" (concatLists [["${head parts}-${ty}"] (tail parts)]))
                      pkg-flake;

        builds = {
          cpu     = build-flake "cpu";
          cuda-10 = build-flake "cuda-10";
          cuda-11 = build-flake "cuda-11";
        };

        extra-packages = {
          packages = {
            haddocks-join = (pkgset.cpu.pkgs.callPackage ./nix/haddock-combine.nix {}) {
              hsdocs = [
                builds.cpu.packages."libtorch-ffi-cpu:lib:libtorch-ffi".doc
                builds.cpu.packages."libtorch-ffi-helper-cpu:lib:libtorch-ffi-helper".doc
                builds.cpu.packages."hasktorch-cpu:lib:hasktorch".doc
                builds.cpu.packages."hasktorch-gradually-typed-cpu:lib:hasktorch-gradually-typed".doc
              ];
            };
          };
        };
        packages = with builds;
          builtins.foldl' (sum: v: lib.recursiveUpdate sum v) {} (
            if system == "x86_64-darwin"
            then  [cpu extra-packages]
            else  [cpu cuda-10 cuda-11 extra-packages]
          );
          

      in with builds;
         packages // (
          let
            dev = with pkgset;
              if !build-config.dev.cudaSupport then cpu else
              if build-config.dev.cudaMajorVersion == "10" then cuda-10 else cuda-11;
          in {
            lib = pkgset;
            devShell =  dev.pkgs.callPackage ./shell.nix {
              inherit (build-config.dev) cudaSupport cudaMajorVersion;
            };
          } )
    ));
}
