{ compilerVersion ? "ghc843" }:
let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              # type-combinators = pkgs.callPackage ./vendor/type-combinators.nix { };

              hasktorch-codegen            = haskellPackagesNew.callPackage ./codegen { };
              # hasktorch-core             = haskellPackagesNew.callPackage ./core { };
              # hasktorch-indef            = haskellPackagesNew.callPackage ./indef { };
              # hasktorch-signatures       = haskellPackagesNew.callPackage ./signatures { };
              # hasktorch-signatures-types = haskellPackagesNew.callPackage ./signatures/types { };
              # hasktorch-partial          = haskellPackagesNew.callPackage ./signatures/partial { };
              # hasktorch-types-th         = haskellPackagesNew.callPackage ./types/th { };
              # hasktorch-types-thc        = haskellPackagesNew.callPackage ./types/thc { };
              #hasktorch-raw-th           = haskellPackagesNew.callPackage ./raw/th { };
              #hasktorch-raw-thc          = haskellPackagesNew.callPackage ./raw/thc { };
              #hasktorch-raw-tests        = haskellPackagesNew.callPackage ./raw/tests { };
              # hasktorch-examples         = haskellPackagesNew.callPackage ./examples { };
            };
          };
        };
      };
    };
  };
  # pkgs = import (fetchGit (import ./version.nix)) { inherit config; };
  pkgs = import <nixpkgs> { inherit config; };
in
  {
    hasktorch-aten = pkgs.callPackage ./vendor/aten.nix { };
    hasktorch-codegen = pkgs.haskell.packages.${compilerVersion}.hasktorch-codegen;
    # type-combinators = pkgs.haskell.packages.${compilerVersion}.type-combinators;
    # hasktorch-raw-th = pkgs.haskell.packages.${compilerVersion}.hasktorch-raw-th;
  }

