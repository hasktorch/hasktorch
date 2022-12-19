{
  description = "Hasktorch";

  nixConfig = {
    substituters = [
      "https://cache.nixos.org"
    ];
    extra-substituters = [
      "https://cache.iog.io"
      "https://hasktorch.cachix.org"
    ];
    extra-trusted-public-keys = [
      "hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ="
      "hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc="
    ];
    allow-import-from-derivation = "true";
    bash-prompt = "\\[\\033[1m\\][dev-hasktorch$(__git_ps1 \" (%s)\")]\\[\\033\[m\\]\\040\\w$\\040";
    
  };

  inputs = {
    haskell-nix = {
      url = "github:input-output-hk/haskell.nix?rev=ec0c59e2de05053c21301bc959a27df92fe93376";
    };
    nixpkgs.follows = "haskell-nix/nixpkgs-unstable";
    utils.follows = "haskell-nix/flake-utils";
    iohkNix = {
      url = "github:input-output-hk/iohk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    tokenizers = {
      url = "github:hasktorch/tokenizers/flakes";
      inputs.utils.follows = "haskell-nix/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  };

  outputs = { self
            , nixpkgs
            , haskell-nix
            , utils
            , iohkNix
            , tokenizers
            , pre-commit-hooks
            , ... }: with utils.lib;
    let
      inherit (nixpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      build-config.dev     = import ./nix/dev-config.nix; # only used to generate devshell
      build-config.cpu     = { profiling = true; cudaSupport = false; cudaMajorVersion = "invalid"; };
      build-config.cuda-10 = { profiling = true; cudaSupport = true;  cudaMajorVersion = "10"; };
      build-config.cuda-11 = { profiling = true; cudaSupport = true;  cudaMajorVersion = "11"; };

      hasktorch.overlays = {
        cpu-config     = final: prev: { hasktorch-config = build-config.cpu; };
        cuda-10-config = final: prev: { hasktorch-config = build-config.cuda-10; };
        cuda-11-config = final: prev: { hasktorch-config = build-config.cuda-11; };

        dev-tools = final: prev: {
          haskell-nix = prev.haskell-nix // {
            custom-tools = prev.haskell-nix.custom-tools // (prev.callPackage ./nix/haskell-language-server {});
          };
        };

        hasktorch-project = ty: final: prev:
          let libtorch = prev.pkgs.callPackage ./nix/libtorch.nix {
                cudaSupport = build-config."${ty}".cudaSupport;
                device = ty;
              };
              libtorch-libs = {
                torch = libtorch;
                c10 = libtorch;
                torch_cpu = libtorch;
              } // (if build-config."${ty}".cudaSupport then {
                torch_cuda = libtorch;
              } else {
              });
          in {
            hasktorchProject = import ./nix/haskell.nix ({
              pkgs = prev // libtorch-libs;
              compiler-nix-name = "ghc924";
              inherit (prev) lib;
              profiling = build-config."${ty}".profiling;
              cudaSupport = build-config."${ty}".cudaSupport;
            });
          } // libtorch-libs;
      };

      generic-pkgset = system: ty:
        let
          overlays = [
            haskell-nix.overlay
            iohkNix.overlays.haskell-nix-extra
            tokenizers.overlay
            hasktorch.overlays.dev-tools
            (hasktorch.overlays.hasktorch-project ty)
          ];
        in
          { pkgs = import nixpkgs { inherit system overlays; inherit (haskell-nix) config;};
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

        ghc = pkgset.cpu.pkgs.hasktorchProject.ghcWithPackages (_: []);

        extra-packages = {
          packages = {
            haddocks-join = (pkgset.cpu.pkgs.callPackage ./nix/haddock-combine.nix {
              inherit ghc;
            }) {
              hsdocs = [
                builds.cpu.packages."libtorch-ffi-cpu:lib:libtorch-ffi".doc
                builds.cpu.packages."libtorch-ffi-helper-cpu:lib:libtorch-ffi-helper".doc
                builds.cpu.packages."hasktorch-cpu:lib:hasktorch".doc
                builds.cpu.packages."hasktorch-gradually-typed-cpu:lib:hasktorch-gradually-typed".doc
              ];
            };
          };
          checks = {
            # pre-commit-check = pre-commit-hooks.lib.${system}.run {
            #   src = ./.;
            #   hooks = {
            #     nixpkgs-fmt = {
            #       enable = true;
            #       excludes = [
            #         "^nix/sources\.nix"
            #       ];
            #     };
            #     ormolu = {
            #       enable = true;
            #       excludes = [
            #         "^Setup.hs$"
            #         "^libtorch-ffi/.*$"
            #       ];
            #     };
            #   };
            # };
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
            devShells.default =  dev.pkgs.callPackage ./shell.nix {
              # preCommitShellHook = self.checks.${system}.pre-commit-check.shellHook;
              inherit (build-config.dev) cudaSupport cudaMajorVersion;
            };
          } )
    ));
}
