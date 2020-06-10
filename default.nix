let
  haskellNix = import (builtins.fetchTarball https://github.com/input-output-hk/haskell.nix/archive/master.tar.gz) {};

  nixpkgsSrc = haskellNix.sources.nixpkgs-2003;

  libtorchSrc = pkgs:
    let src = pkgs.fetchFromGitHub {
          owner  = "stites";
          repo   = "pytorch-world";
          rev    = "6dc929a791918fff065bb40cbc3db8a62beb2a30";
          sha256 = "140a2l1l1qnf7v2s1lblrr02mc0knsqpi06f25xj3qchpawbjd4c";
    };
    in (pkgs.callPackage "${src}/libtorch/release.nix" { });

  libtorchOverlay = pkgsNew: pkgsOld: {
    inherit (libtorchSrc pkgsOld)
      libtorch_cpu
      libtorch_cudatoolkit_9_2
      libtorch_cudatoolkit_10_2
    ;
    # c10 = pkgsNew.libtorch_cpu;
  };

  nixpkgsArgs = haskellNix.nixpkgsArgs // { overlays = haskellNix.overlays ++ [ libtorchOverlay ]; };

in

{ pkgs ? import nixpkgsSrc nixpkgsArgs
, haskellCompiler ? "ghc883"
}:

let

  # ghcide = (import sources.ghcide-nix {})."ghcide-${haskellCompiler}";

  # 'cabalProject' generates a package set based on a cabal.project (and the corresponding .cabal files)
  hsPkgs = let pkgsNew = pkgs // { overlays = pkgs.overlays ++ [ (pkgsNew: pkgsOld: { c10 = pkgsOld.libtorch_cpu; }) ]; }; in
    pkgsNew.haskell-nix.cabalProject {
      # 'cleanGit' cleans a source directory based on the files known by git
      src = pkgsNew.haskell-nix.haskellLib.cleanGit { name = "hasktorch"; src = ./.; };
      compiler-nix-name = haskellCompiler;
      # pkg-def-extras = [{
      #   packages = { "c10" = pkgs.libtorch_cpu; };
      # }];
      modules = [
        ({ config, ... }: {
          packages.libtorch-ffi.configureFlags = [ "--extra-include-dirs=${pkgsNew."libtorch_cpu"}/include/torch/csrc/api/include" ];
        })
      ];
      # modules = [{
      #   # Specific package overrides would go here for example:
      #   packages.cbors.package.ghcOptions = "-Werror";
      #   packages.cbors.patches = [ ./one.patch ];
      #   packages.cbors.flags.optimize-gmp = false;
      #   # It may be better to set flags in `cabal.project` instead
      #   # (`plan-to-nix` will include them as defaults).
      # }];
    };

  shell = hsPkgs.shellFor {
    withHoogle = true;
    buildInputs = [
      pkgs.cabal-install
      pkgs.haskellPackages.ghcid
      pkgs.haskellPackages.hpack
      pkgs.haskellPackages.brittany
      pkgs.haskellPackages.dhall
      # ghcide
    ];
  };

in
  rec {
    inherit hsPkgs;

    codegen = hsPkgs.codegen;

    libtorch-ffi_cpu = hsPkgs.libtorch-ffi;
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
