{ lib
, stdenv
, pkgs
, haskell-nix
, buildPackages
, config ? {}
# GHC attribute name
, compiler ? config.haskellNix.compiler or "ghc883"
# Enable profiling
, profiling ? config.haskellNix.profiling or false
# Version info, to be passed when not building from a git work tree
, gitrev ? null
# Enable CUDA support
, cudaSupport ? false
# Add packages on top of the package set derived from cabal resolution
, extras ? (_: {})
}:

let

  src = haskell-nix.haskellLib.cleanGit {
      name = "hasktorch";
      src = ../.;
  };

  projectPackages = lib.attrNames (haskell-nix.haskellLib.selectProjectPackages
    (haskell-nix.cabalProject { inherit src; }));

  setupNumCores = libname: ''
      case "$(uname)" in
        "Darwin")
            TOTAL_MEM_GB=`sysctl hw.physmem | awk '{print int($2/1024/1024/1024)}'`
            NUM_CPU=$(sysctl -n hw.ncpu)
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

  # This creates the Haskell package set.
  # https://input-output-hk.github.io/haskell.nix/user-guide/projects/
  pkgSet = haskell-nix.cabalProject {
    inherit src;

    compiler-nix-name = compiler;

    # these extras will provide additional packages
    # on top of the package set derived from cabal resolution.
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
      # TODO: Compile all local packages with -Werror:
      # {
      #   packages = lib.genAttrs projectPackages
      #     (name: { configureFlags = [ "--ghc-option=-Werror" ]; });
      # }

      # Enable profiling
      (lib.optionalAttrs profiling {
        enableLibraryProfiling = true;
        packages.examples.components.all.enableExecutableProfiling = true;
      })

      # Add non-Haskell dependencies
      {
        packages.libtorch-ffi = {
          preConfigure = setupNumCores "libtorch-ffi";
          configureFlags = [
            "--extra-lib-dirs=${pkgs.torch}/lib"
            "--extra-include-dirs=${pkgs.torch}/include"
            "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
          ];
          flags = {
            cuda = cudaSupport;
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
        packages.cryptonite-openssl.package.identifier.name = "cryptonite-openssl";

        # Some tests don't work on every platform
        # packages.hasktorch.components.all.platforms =
        #   with stdenv.lib.platforms; lib.mkForce [ linux darwin windows ];
        # packages.hasktorch.components.tests.all.platforms =
        #   with stdenv.lib.platforms; lib.mkForce [ linux darwin ];

        # Disable doctests for now
        # TODO: see if we can turn these on again (waiting for https://github.com/input-output-hk/haskell.nix/pull/427)
        packages.codegen.components.tests.doctests.buildable = lib.mkForce false;
        packages.codegen.components.tests.doctests.doCheck = false;
        packages.hasktorch.components.tests.doctests.buildable = lib.mkForce false;
        packages.hasktorch.components.tests.doctests.doCheck = false;
        # Disable cabal-doctest tests by turning off custom setups
        packages.hasktorch.package.buildType = lib.mkForce "Simple";
      }

      # Stamp packages with the git revision
      {
        # packages.codegen.components.exes.codegen.postInstall = setGitRev;
        # packages.examples.components.exes.examples.postInstall = setGitRev;
        # Work around Haskell.nix issue when setting postInstall on components
        packages.codegen.components.all.postInstall = lib.mkForce setGitRev;
        packages.libtorch-ffi.components.all.postInstall = lib.mkForce setGitRev;
        packages.libtorch-ffi-helper.components.all.postInstall = lib.mkForce setGitRev;
        packages.hasktorch.components.all.postInstall = lib.mkForce setGitRev;
        packages.examples.components.all.postInstall = lib.mkForce setGitRev;
        packages.experimental.components.all.postInstall = lib.mkForce setGitRev;
      }
    ];
  };

  # setGitRev is a postInstall script to stamp executables with
  # version info. It uses the "gitrev" argument, if set. Otherwise,
  # the revision is sourced from the local git work tree.
  setGitRev = ''
    ${haskellBuildUtils}/bin/set-git-rev "${gitrev'}" $out/bin/* || true
  '';
  gitrev' = if (gitrev == null)
    then buildPackages.commonLib.commitIdFromGitRepoOrZero ../.git
    else gitrev;
  haskellBuildUtils = buildPackages.haskellBuildUtils.package;

in
  pkgSet
