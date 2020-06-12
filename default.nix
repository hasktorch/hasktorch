args@{ cudaVersion ? null, ... }:

let
  shared = (import ./shared.nix { inherit (args); });
  default = (
    if cudaVersion == null
    then shared.defaultCpu
    else if cudaVersion == 9
      then shared.defaultCuda92
      else shared.defaultCuda102
  );
  pkgs = default.pkgs;
  hsPkgs = default.hsPkgs;
  getComponents = group: name:
    pkgs.haskell-nix.haskellLib.collectComponents group (package: (builtins.tryEval (package.identifier.name == name)).value) hsPkgs;
in
  assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != { }) || (cudaVersion == 10 && shared.defaultCuda102 != { });
  {
    inherit getComponents;
    codegen = getComponents "exes" "codegen";
    libtorch-ffi = getComponents "library" "libtorch-ffi";
    hasktorch = getComponents "library" "hasktorch";
    examples = getComponents "exes" "examples";
    experimental = getComponents "exes" "experimental";
  }
