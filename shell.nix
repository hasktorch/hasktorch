{ cudaSupport ? false, mklSupport ? true, cuda_path ? null }:

let
  release = import ./build.nix { inherit cudaSupport mklSupport; };
  pkgs = import <nixpkgs> {};
  isNixOS = (builtins.tryEval (import <nixos> {})).success;
  lib = pkgs.lib;
  cuda_driver_path = folder:
    let snoc = p: "${p}/${folder}";
    in
      if cuda_path != null
      then snoc cuda_path
      else
        if isNixOS
        then snoc release.nvidia_x11-pinned
        else (lib.strings.concatStringsSep ":"
          # include edgecases here
          (if (folder == "include")
          then []
          else ["/usr/lib/x86_64-linux-gnu"]));
in

with release;

pkgs.mkShell {
  name = "hasktorch-dev-environment";
  PATH               = "${cudatoolkit}/bin";
  LD_LIBRARY_PATH    = "${hasktorch-aten}/lib:/lib64:${cuda_driver_path "lib"}";
  C_INCLUDE_PATH     = "${hasktorch-aten}/include:/include:${cuda_driver_path "include"}";
  CPLUS_INCLUDE_PATH = "${hasktorch-aten}/include:/include:${cuda_driver_path "include"}";

  # FIXME: this might be cleaner than the above
  # runtimeDependencies = [
  #   cudatoolkit
  #   hasktorch-aten
  # ];

  buildInputs = [
    # pkgs.busybox
    pkgs.cabal-install
    (pkgs.haskellPackages.ghcWithPackages (self: with self; [ hspec-discover ]))

    hasktorch-aten
    hasktorch-codegen
    hasktorch-types-th
    hasktorch-ffi-th
    # hasktorch-ffi-thc will have mismatched cuda versions depending on if you build on nixos or not. Best to just let cabal handle them
    ] ++ (lib.optionals cudaSupport [ hasktorch-types-thc ]); # hasktorch-ffi-thc ]);
}

