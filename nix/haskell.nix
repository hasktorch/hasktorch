{ pkgs
, lib
, compiler-nix-name
, profiling
, cudaSupport
, extras ? (_: {})
, src ? (pkgs.haskell-nix.haskellLib.cleanGit {
      name = "hasktorch";
      src = ../.;
  })
, projectPackages ? lib.attrNames (pkgs.haskell-nix.haskellLib.selectProjectPackages
    (pkgs.haskell-nix.cabalProject' {
      inherit src compiler-nix-name;
    }).hsPkgs)
}:
let
  inherit (pkgs) stdenv;

  setupNumCores = libname: ''
      case "$(uname)" in
        "Darwin")
            TOTAL_MEM_GB=`${pkgs.procps}/bin/sysctl hw.physmem | awk '{print int($2/1024/1024/1024)}'`
            NUM_CPU=$(${pkgs.procps}/bin/sysctl -n hw.ncpu)
          ;;
        "Linux")
            TOTAL_MEM_GB=`grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'`
            NUM_CPU=$(nproc --all)
          ;;
      esac

      USED_MEM_GB=`echo $TOTAL_MEM_GB | awk '{print int(($1 + 1) / 2)}'`
      USED_NUM_CPU=`echo $NUM_CPU | awk '{print int(($1 + 1) / 2)}'`
      USED_NUM_CPU=`echo $USED_MEM_GB $USED_NUM_CPU | awk '{if($1<x$2) {print $1} else {print $2}}'`
      USED_MEM_GB=`echo $USED_NUM_CPU | awk '{print ($1)"G"}'`
      USED_MEMX2_GB=`echo $USED_NUM_CPU | awk '{print ($1 * 2)"G"}'`
      if [ "${libname}" = "hasktorch " ] ; then
        export NIX_BUILD_CORES=$USED_NUM_CPU
        sed -i -e 's/\(^\(.*\)default-extension\)/\2ghc-options: -j'$USED_NUM_CPU' +RTS -A128m -n2m -M'$USED_MEMX2_GB' -RTS\n\1/g' ${libname}.cabal
      else
        sed -i -e 's/\(^\(.*\)default-extension\)/\2ghc-options: -j'$USED_NUM_CPU' +RTS -A128m -n2m -M'$USED_MEM_GB' -RTS\n\1/g' ${libname}.cabal
      fi
    '';

  pkgSet = pkgs.haskell-nix.cabalProject' {
    inherit src compiler-nix-name;

    pkg-def-extras = [
      extras
      (hackage: {
        packages = {
          # see https://github.com/well-typed/cborg/issues/242
          "primitive" = (((hackage.primitive)."0.7.0.0").revisions).default;
        };
      })
    ];

    modules = [
      # Enable profiling
      (lib.optionalAttrs profiling {
        enableLibraryProfiling = true;
        packages.examples.enableExecutableProfiling = true;
        packages.experimental.enableExecutableProfiling = true;
      })

      # Add non-Haskell dependencies
      {
        packages.tokenizers = {
          configureFlags = [
            "--extra-lib-dirs=${pkgs.tokenizers_haskell}/lib"
          ];
        };
        packages.libtorch-ffi = {
          preConfigure = setupNumCores "libtorch-ffi";
          configureFlags = [
            "--extra-lib-dirs=${pkgs.torch}/lib"
            "--extra-include-dirs=${pkgs.torch}/include"
            "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
          ];
          flags = {
            cuda = cudaSupport;
            rocm = false;
            gcc = !cudaSupport && pkgs.stdenv.hostPlatform.isDarwin;
          };
        };
        packages.hasktorch = {
          preConfigure = setupNumCores "hasktorch";
        };
      }

      # Misc. build fixes for dependencies
      {
        # Some packages are missing identifier.name
        # packages.cryptonite-openssl.package.identifier.name = "cryptonite-openssl";

        # Disable doctests for now
        # TODO: see if we can turn these on again (waiting for https://github.com/input-output-hk/haskell.nix/pull/427)
        #packages.codegen.components.tests.doctests.buildable = lib.mkForce false;
        #packages.codegen.components.tests.doctests.doCheck = false;
        #packages.hasktorch.components.tests.doctests.buildable = lib.mkForce false;
        #packages.hasktorch.components.tests.doctests.doCheck = false;
        # Disable cabal-doctest tests by turning off custom setups
        #packages.hasktorch.package.buildType = lib.mkForce "Simple";
      }
    ];
  };
in
  pkgSet // {
    inherit projectPackages;
  }
