############################################################################
# Builds Haskell packages with Haskell.nix
############################################################################

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
}:

let

  src = haskell-nix.haskellLib.cleanGit {
      name = "hasktorch";
      src = ../.;
  };

  projectPackages = lib.attrNames (haskell-nix.haskellLib.selectProjectPackages
    (haskell-nix.cabalProject { inherit src; }));

  # This creates the Haskell package set.
  # https://input-output-hk.github.io/haskell.nix/user-guide/projects/
  pkgSet = haskell-nix.cabalProject {
    inherit src;

    compiler-nix-name = compiler;

    # these extras will provide additional packages
    # ontop of the package set derived from cabal resolution.
    pkg-def-extras = [(hackage: {
      packages = {
          # Win32 = hackage.Win32."2.8.3.0".revisions.default;
      };
    })];

    modules = [
      # { compiler.nix-name = compiler; }

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

      # Add dependencies
      {
        packages.libtorch-ffi = {
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
      }

      # Misc. build fixes for dependencies  
      {
        # Some packages are missing identifier.name
        packages.cryptonite-openssl.package.identifier.name = "cryptonite-openssl";

        # Disable cabal-doctest tests by turning off custom setups
        # packages.hasktorch.package.buildType = lib.mkForce "Simple";

        # Some tests don't work on every platform
        # packages.hasktorch.components.all.platforms =
        #   with stdenv.lib.platforms; lib.mkForce [ linux darwin windows ];
        # packages.hasktorch.components.tests.all.platforms =
        #   with stdenv.lib.platforms; lib.mkForce [ linux darwin ];

        # Turn off doctests
        packages.codegen.components.tests.doctests.doCheck = false;
      }

      {
        packages.examples.components.all.postInstall = setGitRev; # lib.mkForce setGitRev;
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