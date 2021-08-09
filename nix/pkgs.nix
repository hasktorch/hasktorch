{ compiler-nix-name ? "ghc8105"
, gitrev
, lib
#, stdenv
#, pkgs
, haskell-nix
#, buildPackages
#, config
#, cudaSupport
#, extras
}:
# Hasktorch packages overlay
pkgs: _: with pkgs; {

  hasktorchProject = import ./haskell.nix {
    inherit
      lib
      pkgs
      haskell-nix
      #buildPackages
      #config
      #cudaSupport
      #extras
      compiler-nix-name
      gitrev
      ;
  };
}
