final: prev: let
  pythonPackagesExtensions =
    prev.pythonPackagesExtensions
    ++ [
      (
        python-final: python-prev: {
          torch = python-final.torch.overridePythonAttrs (old: rec {
            version = "2.0.0.0";
            src = prev.fetchFromGitHub {
              owner = "pytorch";
              repo = "pytorch";
              rev = "refs/tags/v${version}";
              fetchSubmodules = true;
              hash = "sha256-xUj77yKz3IQ3gd/G32pI4OhL3LoN1zS7eFg0/0nZp5I=";
            };
          });
        }
      )
    ];
  inherit (final.python3Packages) torch;
in {
  libtorch-bin = torch;
  libtorch = torch;
  torch = torch;
  c10 = torch;
  torch_cpu = torch;
  torch_cuda = torch;

  datasets = final.callPackage ./datasets.nix {};
  hasktorch = final.callPackage ./package.nix {};
}
