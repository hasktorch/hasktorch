{ withCuda ? true }:

if withCuda
  then (import ../nix/shared.nix { }).shell-hasktorch_cudatoolkit_10_1
  else (import ../nix/shared.nix { }).shell-hasktorch_cpu
