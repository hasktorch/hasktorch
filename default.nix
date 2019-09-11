let
  shared = import ./nix/shared.nix { };
in
  { inherit (shared)
      libtorch-ffi
      hasktorch
      hasktorch-codegen
      hasktorch-examples
    ;
  }
