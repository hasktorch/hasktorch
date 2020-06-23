############################################################################
# Hasktorch Nix build
#
# TODO: document top-level attributes and how to build them
#
############################################################################

{ system ? builtins.currentSystem
, crossSystem ? null
# allows to customize ghc and profiling (see ./nix/haskell.nix):
, config ? {}
# allows to override dependencies of the hasktorch project without modifications
, sourcesOverride ? {}
# if true, activates CUDA support
, cudaSupport ? false
# if cudaSupport is true, this needs to be set to a valid CUDA major version number, e.g. 10:
# nix-build --arg cudaSupport true --argstr cudaMajorVersion 10
, cudaMajorVersion ? null
# pinned version of nixpkgs augmented with various overlays.
, pkgs ? import ./nix/default.nix { inherit system crossSystem config sourcesOverride cudaSupport cudaMajorVersion; }
# git sha1 hash, to be passed when not building from a git work tree.
, gitrev ? null
}:

# commonLib includes util.nix and nixpkgs lib.
with pkgs; with commonLib;

let

  haskellPackages = recRecurseIntoAttrs
    # the Haskell.nix package set, reduced to local packages.
    (selectProjectPackages hasktorchHaskellPackages);

  libs = collectComponents' "library" haskellPackages;
  exes = collectComponents' "exes" haskellPackages;

  self = {
    # Inherit haskellPackages so that you can still access things that are not exposed below.
    inherit haskellPackages;

    inherit (haskellPackages.hasktorch.identifier) version;

    # `tests` are the test suites which have been built.
    tests = collectComponents' "tests" haskellPackages;
    # `benchmarks` (only built, not run).
    benchmarks = collectComponents' "benchmarks" haskellPackages;

    # Grab hasktorch's library components.
    inherit (libs)
      libtorch-ffi
      libtorch-ffi-helper
      hasktorch
      ;

    # Grab hasktorch's executable components.
    inherit (exes)
      codegen
      examples
      experimental
      ;

    checks = recurseIntoAttrs {
      # `checks.tests` collect results of executing the tests:
      tests = collectChecks haskellPackages;
      # Example of a linting script used by Buildkite.
      # lint-fuzz = callPackage ./nix/check-lint-fuzz.nix {};
    };

    shell = import ./shell.nix {
      inherit pkgs;
      withHoogle = true;
    };

    # Building the stack shell doesn't work in the sandbox. Pass `--option sandbox relaxed` or
    # `--option sandbox false` to be able to build this. You have to be root in order to that.
    # stackShell = import ./nix/stack-shell.nix {
    #   inherit pkgs;
    # };

    # Build the documentation.
    combined-haddock = let
      haddock-combine = callPackage ./nix/haddock-combine.nix {
        runCommand = runCommand;
        lib = lib;
        ghc = hasktorchHaskellPackages.ghcWithPackages (ps: []);
      };
      in haddock-combine {
        hspkgs = builtins.attrValues libs;
        prologue = writeTextFile {
          name = "prologue";
          text = "Documentation for hasktorch and its libraries.";
        };
      };
  };

in

  self