{ compilerVersion ? "ghc844", cudaSupport ? false, mklSupport ? true }:
let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: let

      # cudatoolkit_9_0 must use gcc < 6
      stdenv5 = pkgs.overrideCC pkgs.stdenv pkgs.gcc5;
      gfortran-gcc5 = pkgs.wrapCC (pkgs.gcc5.cc.override {
          name = "gfortran-gcc5";
          langFortran = true;
          langCC = false;
          langC = false;
          langObjC = false;
          langObjCpp = false;
          profiledCompiler = false;
      });

    in rec {
      cmake_3_12_2 = pkgs.libsForQt5.callPackage ./ffi/deps/cmake-nix-3_12_2 {};
      magma = (pkgs.callPackage ./ffi/deps/magma-nix {}).magma.override {
        mklSupport = true;
        cudatoolkit = pkgs.cudatoolkit_9_0;
        cmake = cmake_3_12_2;
        stdenv = stdenv5;
        gfortran = gfortran-gcc5;
      };

      hasktorch-aten =
        pkgs.callPackage ./ffi/deps/hasktorch-aten.nix {
          inherit (pkgs.python36Packages) typing pyaml;
          inherit cudaSupport mklSupport;
          magma_cudatoolkit_9_0 = magma;
          stdenv = stdenv5;
          gfortran = gfortran-gcc5;
        };

      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              hasktorch-ffi-th =
                haskellPackagesNew.callPackage ./ffi/ffi/th { ATen = hasktorch-aten; };
              # hasktorch-ffi-thc =
              #   haskellPackagesNew.callPackage ./ffi/ffi/thc { ATen = hasktorch-aten; };
              hasktorch-ffi-tests =
                haskellPackagesNew.callPackage ./ffi/ffi/tests { };

              hasktorch-codegen =
                haskellPackagesOld.callPackage ./ffi/codegen { };

              hasktorch-types-th =
                haskellPackagesOld.callPackage ./ffi/types/th { };
              # hasktorch-types-thc =
              #   haskellPackagesNew.callPackage ./ffi/types/thc { };

              # hasktorch-signatures =
              #   haskellPackagesNew.callPackage ./signatures { };
              # hasktorch-signatures-types =
              #   haskellPackagesNew.callPackage ./signatures/types { };
              # hasktorch-signatures-partial =
              #   haskellPackagesNew.callPackage ./signatures/partial { };

              # hasktorch-indef =
              #   haskellPackagesNew.callPackage ./indef { };
              # hasktorch-core =
              #   haskellPackagesNew.callPackage ./core { };

              # hasktorch-examples =
              #   haskellPackagesNew.callPackage ./examples { };
            };
          };
        };
      };
    };
  };
  pkgs = import <nixpkgs> { inherit config; };
  ghc = pkgs.haskell.packages.${compilerVersion};

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
    # hasktorch-ffi-thc
    hasktorch-ffi-tests
    hasktorch-types-th;
    # hasktorch-types-thc;
}
