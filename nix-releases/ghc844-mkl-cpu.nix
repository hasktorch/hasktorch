{ cudaSupport ? false, mklSupport ? true }:

(import ../build.nix { inherit cudaSupport mklSupport; compilerVersion = "ghc844"; })

