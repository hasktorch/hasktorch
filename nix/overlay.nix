final: prev: let
  inherit (final) lib;
  inherit (final.python3Packages) torch;
  inherit (prev.haskell.lib) unmarkBroken overrideCabal;
  c10 = torch;
  torch_cpu = torch;
  ghcName = "ghc928";
in {
  pythonPackagesExtensions =
    prev.pythonPackagesExtensions
    ++ [
      (
        python-final: python-prev: {
          torch = python-prev.torch.overridePythonAttrs (old: rec {
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

  haskell = lib.recursiveUpdate prev.haskell {
    packages.${ghcName} = prev.haskell.packages.${ghcName}.override {
      overrides = hfinal: hprev: {
        constraints-deriving = unmarkBroken hprev.constraints-deriving;
        libtorch-ffi-helper = hprev.callCabal2nix "libtorch-ffi-helper" ../libtorch-ffi-helper {};
        libtorch-ffi = overrideCabal (hprev.callCabal2nix "libtorch-ffi" ../libtorch-ffi {inherit torch c10 torch_cpu;}) (_: {
          enableLibraryProfiling = false;
          configureFlags = [
            "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include"
          ];
        });
        codegen = hprev.callCabal2nix "codegen" ../codegen {};
        hasktorch =
          overrideCabal (hprev.callCabal2nix "hasktorch" ../hasktorch {})
          (_: {
            # Expecting data for tests
            doCheck = false;
            enableLibraryProfiling = false;
          });
      };
    };
  };
}
