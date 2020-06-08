let 
  # Fetch the latest haskell.nix and import its default.nix
  haskellNix = import (builtins.fetchTarball https://github.com/input-output-hk/haskell.nix/archive/master.tar.gz) {};

  # haskell.nix provides access to the nixpkgs pins which are used by our CI, hence
  # you will be more likely to get cache hits when using these.
  # But you can also just use your own, e.g. '<nixpkgs>'
  nixpkgsSrc = haskellNix.sources.nixpkgs-2003;

  # haskell.nix provides some arguments to be passed to nixpkgs, including some patches
  # and also the haskell.nix functionality itself as an overlay.
  nixpkgsArgs = haskellNix.nixpkgsArgs;
in
{ pkgs ? import nixpkgsSrc nixpkgsArgs
, haskellCompiler ? "ghc8101"
}:

# 'cabalProject' generates a package set based on a cabal.project (and the corresponding .cabal files)
pkgs.haskell-nix.cabalProject {
  # 'cleanGit' cleans a source directory based on the files known by git
  src = pkgs.haskell-nix.haskellLib.cleanGit { name = "hasktorch"; src = ./.; };
  compiler-nix-name = haskellCompiler;
  # pkg-def-extras = [
  #   # Additional packages ontop of all those listed in `cabal.project`
  # ];
  # modules = [{
  #   # Specific package overrides would go here for example:
  #   packages.cbors.package.ghcOptions = "-Werror";
  #   packages.cbors.patches = [ ./one.patch ];
  #   packages.cbors.flags.optimize-gmp = false;
  #   # It may be better to set flags in `cabal.project` instead
  #   # (`plan-to-nix` will include them as defaults).
  # }];
}


# let
#   shared = import ./nix/shared.nix { };
# in
#   { inherit (shared)
#       hasktorch-codegen
#       libtorch-ffi_cpu
#       hasktorch_cpu
#       hasktorch-examples_cpu
#       hasktorch-experimental_cpu
#       hasktorch-docs
#     ;

#     ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_9_2"} = shared.libtorch-ffi_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch_cudatoolkit_9_2"} = shared.hasktorch_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_9_2"} = shared.hasktorch-examples_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_9_2"} = shared.hasktorch-experimental_cudatoolkit_9_2;

#     ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_10_2"} = shared.libtorch-ffi_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch_cudatoolkit_10_2"} = shared.hasktorch_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_10_2"} = shared.hasktorch-examples_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_10_2"} = shared.hasktorch-experimental_cudatoolkit_10_2;
#   }
