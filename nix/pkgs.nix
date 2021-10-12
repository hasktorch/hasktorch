{ compiler-nix-name
, lib
, haskell-nix
}:
pkgs: _: with pkgs; {
  hasktorchProject = import ./haskell.nix {
    inherit
      lib
      pkgs
      haskell-nix
      compiler-nix-name
      ;
  };
}
