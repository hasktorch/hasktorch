{ cudaSupport ? false, mklSupport ? false }:

(import ./build.nix { inherit cudaSupport mklSupport; }).dev-env
