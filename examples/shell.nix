{ cudaVersion ? null }:
let
  shared = import ../nix/shared.nix { };
in
assert cudaVersion == null || cudaVersion == 9 || cudaVersion == 10;

if cudaVersion == null
then shared.shell-hasktorch-examples_cpu
else if cudaVersion == 9
  then shared.shell-hasktorch-examples_cudatoolkit_9_2
  else shared.shell-hasktorch-examples_cudatoolkit_10_2
