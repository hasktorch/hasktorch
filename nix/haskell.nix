{ pkgs
, lib
, compiler-nix-name
, profiling
, cudaSupport
, extras ? (_: {})
, src ? ../.
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

    pkg-def-extras = [ extras ];

    modules = [
      # Enable profiling
      (lib.optionalAttrs profiling {
        enableLibraryProfiling = true;
#        packages.examples.enableExecutableProfiling = true;
#        packages.hasktorch-gradually-typed.enableExecutableProfiling = true;
      })

      # Fix for "exceptions" build problem with ghc 9.0.1 (https://github.com/input-output-hk/haskell.nix/issues/1177)
      {
        nonReinstallablePkgs = [
          "rts" "ghc-heap" "ghc-prim" "integer-gmp" "integer-simple" "base"
          "deepseq" "array" "ghc-boot-th" "pretty" "template-haskell"
          "ghcjs-prim" "ghcjs-th"
          "ghc-bignum" "exceptions" "stm"
          "ghc-boot"
          "ghc" "Cabal" "Win32" "array" "binary" "bytestring" "containers"
          "directory" "filepath" "ghc-boot" "ghc-compact" "ghc-prim"
          "hpc"
          "mtl" "parsec" "process" "text" "time" "transformers"
          "unix" "xhtml" "terminfo"
        ];
      }

      # Add non-Haskell dependencies
      {
        packages.codegen = {
          preConfigure = setupNumCores "codegen";
        };
        packages.tokenizers = {
          configureFlags = [
            "--extra-lib-dirs=${pkgs.tokenizers-haskell}/lib"
          ];
        };
        packages.libtorch-ffi = {
          preConfigure = setupNumCores "libtorch-ffi";
          configureFlags = [
            "--extra-lib-dirs=${pkgs.torch.out}/lib"
            "--extra-include-dirs=${pkgs.torch.dev}/include"
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
    ];
  };
in
  pkgSet // {
    inherit projectPackages;
  }
