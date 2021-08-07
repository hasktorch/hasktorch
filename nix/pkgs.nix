# Hasktorch packages overlay
final: prev: with final;
  let
    compiler-nix-name = config.haskellNix.compiler or "ghc8105";
  in {
    f
  }






pkgs: _: with pkgs; {
  hasktorchHaskellPackages = import ./haskell.nix {
    inherit
      lib
      stdenv
      pkgs
      haskell-nix
      buildPackages
      config
      gitrev
      cudaSupport
      extras
      ;
  };
}
