{
  description = "Hasktorch";

  nixConfig = {
    extra-substituters = [
      "https://hasktorch.cachix.org"
    ];
    extra-trusted-public-keys = [
      "hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-parts,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      imports = [
        ./nix/nixpkgs-instances.nix
      ];
      flake = {
        overlays.default = import ./nix/overlay.nix;
      };
      perSystem =
        {
          lib,
          pkgs,
          self',
          pkgsCuda,
          ...
        }:
        let
          ghc = "ghc912";
          mnist = (pkgs.callPackage ./nix/datasets.nix { }).mnist;
          mkHasktorchPackageSet =
            t: p:
            lib.mapAttrs' (name: value: lib.nameValuePair "${name}-${t}" value) (
              lib.genAttrs [
                "codegen"
                "hasktorch"
                # Cannot be built until type-level-rewrite-rules supports GHC 9.12
                # "hasktorch-gradually-typed"
                "libtorch-ffi"
                "libtorch-ffi-helper"
              ] (name: p.haskell.packages.${ghc}.${name})
            );
        in
        {
          packages =
            {
              examples = pkgs.haskell.packages.${ghc}.examples;
            }
            // (mkHasktorchPackageSet "cuda" pkgsCuda)
            // (mkHasktorchPackageSet "cpu" pkgs);
          devShells =
            let
              packages =
                p: with p; [
                  codegen
                  examples
                  hasktorch
                  # Cannot be built until type-level-rewrite-rules supports GHC 9.12
                  # hasktorch-gradually-typed
                  libtorch-ffi
                  libtorch-ffi-helper
                ];
              # Developer tools belong here, not in `packages`: shellFor treats
              # `packages` as the set of packages being developed and brings in
              # only their dependencies, so a tool listed there never lands on
              # PATH.
              nativeBuildInputs = p: [
                  p.cabal-install
                  p.haskell.packages.${ghc}.haskell-language-server
                  p.pkg-config
                  (p.python3.withPackages (pp: with pp; [
                    pyyaml
                    typing-extensions
                  ]))
                ]
              ;
              # C libraries that Haskell dependencies link against. `packages`
              # only brings in the *Haskell* dependency closure, so when cabal
              # builds e.g. the `zlib` package from Hackage (which it does
              # whenever the version pinned in cabal.project.freeze differs
              # from the one nixpkgs provides) the C headers must come from
              # here or the configure step fails.
              buildInputs = p: [
                  p.zlib
                  p.zlib.dev
                ]
              ;
            in
            {
              default = self'.devShells.cpu;
              cpu = pkgs.haskell.packages.${ghc}.shellFor {
                inherit packages;
                nativeBuildInputs = nativeBuildInputs pkgs;
                buildInputs = buildInputs pkgs;
              };
              cuda = pkgsCuda.haskell.packages.${ghc}.shellFor {
                inherit packages;
                nativeBuildInputs = nativeBuildInputs pkgsCuda;
                buildInputs = buildInputs pkgsCuda;
              };
            };
          checks = self'.packages;
          apps = {
            codegen-exe = {
              type = "app";
              program = pkgs.writeShellScriptBin "codegen-exe" ''
                ${self'.packages.codegen-cpu}/bin/codegen-exe
              '';
              meta.description = "Code generation executable for libtorch-ffi";
            };
            mnist-mixed-precision = {
              type = "app";
              program = pkgsCuda.writeShellScriptBin "mnist-mixed-precision" ''
                DEVICE=cuda:0 ${self'.packages.examples}/bin/mnist-mixed-precision ${mnist}/
              '';
              meta.description = "A simple mlp implementation of an mnist classifier using untyped tensors";
            };
            static-mnist-cnn = {
              type = "app";
              program = pkgsCuda.writeShellScriptBin "static-mnist-cnn" ''
                DEVICE=cuda:0 ${self'.packages.examples}/bin/static-mnist-cnn ${mnist}/
              '';
              meta.description = "Static Mnist CNN";
            };
            static-mnist-mlp = {
              type = "app";
              program = pkgsCuda.writeShellScriptBin "static-mnist-mlp" ''
                DEVICE=cuda:0 ${self'.packages.examples}/bin/static-mnist-mlp ${mnist}/
              '';
              meta.description = "Static Mnist MLP";
            };
          };
        };
    };
}
