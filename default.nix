############################################################################
# Hasktorch Nix build
#
# TODO: document top-level attributes and how to build them
#
############################################################################

{ system ? builtins.currentSystem
, crossSystem ? null
# allows to cutomize ghc and profiling (see ./nix/haskell.nix):
, config ? {}
# allows to override dependencies of the project without modifications,
# eg. to test build against local checkout of nixpkgs and iohk-nix:
# nix build -f default.nix hasktorch --arg sourcesOverride '{
#   pytorch-world = ../pytorch-world;
#   nixpkgs  = ../nixpkgs;
# }'
, sourcesOverride ? {}
, cudaSupport ? false
, cudaMajorVersion ? null
# pinned version of nixpkgs augmented with various overlays.
, pkgs ? import ./nix/default.nix {
    inherit system crossSystem config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:

# commonLib includes util.nix and nixpkgs lib.
with pkgs; with commonLib;
let

  haskellPackages = recRecurseIntoAttrs
    # the Haskell.nix package set, reduced to local packages.
    (selectProjectPackages hasktorchHaskellPackages);

  self = {
    # Inherit haskellPackages so that you can still access things that are not exposed below.
    inherit haskellPackages;

    inherit (haskellPackages.hasktorch.identifier) version;

    # `tests` are the test suites which have been built.
    tests = collectComponents' "tests" haskellPackages;
    # `benchmarks` (only built, not run).
    benchmarks = collectComponents' "benchmarks" haskellPackages;

    # Grab hasktorch's library components.
    inherit (collectComponents' "library" haskellPackages)
      libtorch-ffi
      libtorch-ffi-helper
      hasktorch
      ;

    # Grab hasktorch's executable components.
    inherit (collectComponents' "exes" haskellPackages)
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

    stackShell = import ./nix/stack-shell.nix {
      inherit pkgs;
    };

    # Attrset of PDF builds of LaTeX documentation.
    # docs = pkgs.callPackage ./docs/default.nix {};
  };
in
  self