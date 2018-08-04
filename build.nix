{ compilerVersion ? "ghc822" }:
let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      hasktorch-aten = pkgs.callPackage ./vendor/aten.nix
                       { inherit (pkgs.python36Packages) typing pyaml; };
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              hasktorch-codegen =
                haskellPackagesOld.callPackage ./codegen { };

              hasktorch-types-th =
                haskellPackagesOld.callPackage ./types/th { };
              hasktorch-types-thc =
                haskellPackagesNew.callPackage ./types/thc { };

              hasktorch-signatures =
                haskellPackagesNew.callPackage ./signatures { };
              hasktorch-signatures-types =
                haskellPackagesNew.callPackage ./signatures/types { };
              hasktorch-partial =
                haskellPackagesNew.callPackage ./signatures/partial { };

              hasktorch-raw-th =
                haskellPackagesNew.callPackage ./raw/th { ATen = hasktorch-aten; };
              hasktorch-raw-thc =
                haskellPackagesNew.callPackage ./raw/thc { ATen = hasktorch-aten; };
              hasktorch-raw-tests =
                haskellPackagesNew.callPackage ./raw/tests { };

              type-combinators =
                haskellPackagesOld.callPackage ./vendor/type-combinators.nix { };

              hasktorch-indef =
                haskellPackagesNew.callPackage ./indef { };
              hasktorch-core =
                haskellPackagesNew.callPackage ./core { };

              hasktorch-examples =
                haskellPackagesNew.callPackage ./examples { };
            };
          };
        };
      };
    };
  };
  pkgs = import <nixpkgs> { inherit config; };
  ghc = pkgs.haskell.packages.${compilerVersion};
in {
  hasktorch-aten = pkgs.hasktorch-aten;
  hasktorch-codegen = ghc.hasktorch-codegen;
  hasktorch-types-th = ghc.hasktorch-types-th;
  hasktorch-raw-th = ghc.hasktorch-raw-th;
  hasktorch-core = ghc.hasktorch-core;
  hasktorch-types-thc = ghc.hasktorch-types-thc;
  hasktorch-raw-thc = ghc.hasktorch-raw-thc;
  hasktorch-signatures = ghc.hasktorch-signatures;
  hasktorch-signatures-types = ghc.hasktorch-signatures-types;
  hasktorch-partial = ghc.hasktorch-partial;
  type-combinators = ghc.type-combinators;
}
