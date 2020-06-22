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
      ;
  };
}