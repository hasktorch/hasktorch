let
  shared = import ./nix/shared.nix { };
in
  { inherit (shared)
      hasktorch_cpu
    ;
  }
