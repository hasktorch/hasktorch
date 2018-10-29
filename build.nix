{ compilerVersion ? "ghc843", cudaSupport ? false, mklSupport ? false }:
let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      hasktorch-aten = pkgs.callPackage ./ffi/deps/aten.nix
                       { inherit (pkgs.python36Packages) typing pyaml; inherit cudaSupport mklSupport; };
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              hasktorch-ffi-th =
                haskellPackagesNew.callPackage ./ffi/ffi/th { ATen = hasktorch-aten; };
              hasktorch-ffi-thc =
                haskellPackagesNew.callPackage ./ffi/ffi/thc { ATen = hasktorch-aten; };
              hasktorch-ffi-tests =
                haskellPackagesNew.callPackage ./ffi/ffi/tests { };

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
              hasktorch-signatures-partial =
                haskellPackagesNew.callPackage ./signatures/partial { };

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
  dev-env = pkgs.stdenv.mkDerivation {
    name = "aten-development-environment";
    buildInputs = [ pkgs.hasktorch-aten ];
  };
in {
  inherit (pkgs) hasktorch-aten;
  inherit
    (ghc)
    # # These dependencies depend on backpack and backpack support in
    # # nix is currently lacking
    # # ref: https://github.com/NixOS/nixpkgs/issues/40128
    # hasktorch-core
    # hasktorch-signatures
    # hasktorch-signatures-partial
    # hasktorch-signatures-types
    # hasktorch-examples
    # hasktorch-indef
    hasktorch-codegen
    hasktorch-ffi-th
    hasktorch-ffi-thc
    hasktorch-types-th
    hasktorch-types-thc;
  inherit dev-env;
}
