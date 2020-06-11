args@{ cudaVersion ? null, ...}:

let
  shared = (import ./shared.nix { inherit (args); });
in
assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != null) || (cudaVersion == 10 && shared.defaultCuda102 != null);

if cudaVersion == null
then if shared.defaultCuda102 == null
  then shared.defaultCpu.hsPkgs
  else shared.defaultCuda102.hsPkgs
else if cudaVersion == 9
  then shared.defaultCuda92.hsPkgs
  else shared.defaultCuda102.hsPkgs
