final: prev:
let
  inherit (final.python3Packages) torch;
in
{
  libtorch-bin = torch;
  libtorch = torch;
  torch = torch;
  c10 = torch;
  torch_cpu = torch;
  torch_cuda = torch;

  datasets = final.callPackage ./datasets.nix { };
  hasktorch = final.callPackage ./package.nix { };
}
