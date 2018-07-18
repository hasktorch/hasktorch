{ compilerVersion ? "ghc843" }:
let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              hasktorch-codegen            = haskellPackagesNew.callPackage ./codegen { };

              hasktorch-types-th         = haskellPackagesNew.callPackage ./types/th { };
              hasktorch-types-thc        = haskellPackagesNew.callPackage ./types/thc { };

              # hasktorch-signatures       = haskellPackagesNew.callPackage ./signatures { };
              # hasktorch-signatures-types = haskellPackagesNew.callPackage ./signatures/types { };
              # hasktorch-partial          = haskellPackagesNew.callPackage ./signatures/partial { };

              # hasktorch-raw-th           = haskellPackagesNew.callPackage ./raw/th { };
              # hasktorch-raw-thc          = haskellPackagesNew.callPackage ./raw/thc { };
              # hasktorch-raw-tests        = haskellPackagesNew.callPackage ./raw/tests { };

              # type-combinators = pkgs.callPackage ./vendor/type-combinators.nix { };

              # hasktorch-indef            = haskellPackagesNew.callPackage ./indef { };
              # hasktorch-core             = haskellPackagesNew.callPackage ./core { };

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
    # hasktorch-aten       = pkgs.callPackage ./vendor/aten.nix { };

    hasktorch-codegen      = pkgs.haskell.packages.${compilerVersion}.hasktorch-codegen;

    hasktorch-types-th     = pkgs.haskell.packages.${compilerVersion}.hasktorch-types-th;
    hasktorch-types-thc    = pkgs.haskell.packages.${compilerVersion}.hasktorch-types-thc;

    # hasktorch-signatures       = pkgs.haskell.packages.${compilerVersion}.hasktorch-signatures;
    # hasktorch-signatures-types = pkgs.haskell.packages.${compilerVersion}.hasktorch-signatures-types;
    # hasktorch-partial          = pkgs.haskell.packages.${compilerVersion}.hasktorch-partial;

    # type-combinators = pkgs.haskell.packages.${compilerVersion}.type-combinators;
    # hasktorch-raw-th = pkgs.haskell.packages.${compilerVersion}.hasktorch-raw-th;
  }

