{ haskell-nix }:

with haskell-nix.haskellLib;
{

  inherit
    selectProjectPackages
    collectComponents';

  inherit (extra)
    collectChecks
    recRecurseIntoAttrs;

}
