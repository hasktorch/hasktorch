final: prev: let
  inherit (final) lib;
  inherit (final.python3Packages) torch;
  inherit (final.haskell.lib) overrideCabal overrideSrc;
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

  haskell =
    prev.haskell
    // {
      packages =
        prev.haskell.packages
        // {
          ${ghcName} = let
            fetchSingletons = name: ''
              ${(prev.fetchFromGitHub {
                owner = "goldfirere";
                repo = "singletons";
                rev = "99bb80b8274a882ebcbbe109c677838404073952";
                sha256 = "sha256-D0q0XcF0wbRBAQxqlc4z44s94ACNZBCWEXZGbq5kghA=";
              })}/${name}'';

            fetchInlineC = name: ''
              ${(prev.fetchFromGitHub {
                owner = "fpco";
                repo = "inline-c";
                rev = "ef87fbae38ed9f646b912f94606d895d0582f1b4";
                sha256 = "sha256-tT/LqUCrVC++N5Mu3eKpK0uXweLg+Qlil5yS9gp1CIE=";
              })}/${name}'';
          in
            prev.haskell.packages.${ghcName}.extend
            (hfinal: hprev: {
              tokenizers_haskell = prev.tokenizersPackages.tokenizers-haskell;
              tokenizers = prev.haskell.lib.dontCheck (hfinal.callCabal2nix "tokenizers"
                ''${prev.fetchFromGitHub {
                    owner = "hasktorch";
                    repo = "tokenizers";
                    rev = "f6e9f989959e0e6e2d9d9ef44204f08f4ea2b523";
                    sha256 = "sha256-9oTk2IFG+iLXelhuXk9C5CkudTVXB/BYBguAkZATL6g=";
                  }}/bindings/haskell/tokenizers-haskell''
                {});
              type-errors-pretty = prev.haskell.lib.doJailbreak (prev.haskell.lib.dontCheck hprev.type-errors-pretty);
              bifunctors = overrideSrc hprev.bifunctors {
                src = prev.fetchFromGitHub {
                  owner = "ekmett";
                  repo = "bifunctors";
                  rev = "4dad73cb3250aa44eb2dc61e3e829939ae97eafc";
                  sha256 = "sha256-s+IpnqFJcIx7Ze2oXVp6b7En01qsYytto96WMnwCGWg=";
                };
              };
              libtorch-ffi-helper = hfinal.callCabal2nix "libtorch-ffi-helper" ../libtorch-ffi-helper {};
              libtorch-ffi = overrideCabal (hfinal.callCabal2nix "libtorch-ffi" ../libtorch-ffi {inherit torch c10 torch_cpu;}) (_: {
                enableLibraryProfiling = false;
                configureFlags = [
                  "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include"
                ];
              });
              th-desugar = overrideSrc hprev.th-desugar {
                src = prev.fetchFromGitHub {
                  owner = "goldfirere";
                  repo = "th-desugar";
                  rev = "b2b9db81d26dc767e7eb4481b7f20cfae6fa0eda";
                  sha256 = "sha256-2qcfzFXO1zgs40uOu+HYLOI7vvFfd2PoVTzGNTjSFmM=";
                };
              };
              inline-c = hfinal.callCabal2nix "inline-c" (fetchInlineC "inline-c") {};
              inline-c-cpp = hfinal.callCabal2nix "inline-c-cpp" (fetchInlineC "inline-c-cpp") {};
              singletons = overrideSrc hprev.singletons {
                src = fetchSingletons "singletons";
              };
              singletons-base = overrideSrc hprev.singletons-base {
                src = fetchSingletons "singletons-base";
              };
              singletons-th = overrideSrc hprev.singletons-th {
                src = fetchSingletons "singletons-th";
              };
              typelevel-rewrite-rules = overrideSrc hprev.typelevel-rewrite-rules {
                src = prev.fetchFromGitHub {
                  owner = "hasktorch";
                  repo = "typelevel-rewrite-rules";
                  rev = "17109219562cb679ee8fe0506f71863add4f23af";
                  sha256 = "sha256-gIRfju/Px+WjEw0FVfgMCjecs8kaUtum7lAwmD022lk=";
                };
              };
              codegen = hfinal.callCabal2nix "codegen" ../codegen {};
              hasktorch-gradually-typed =
                overrideCabal (hfinal.callCabal2nix
                  "hasktorch-gradually-typed"
                  ../experimental/gradually-typed {
                  })
                (_: {
                  # Expecting data for tests
                  doCheck = false;
                  enableLibraryProfiling = false;
                });
              hasktorch =
                overrideCabal (hfinal.callCabal2nix "hasktorch" ../hasktorch {})
                (_: {
                  # Expecting data for tests
                  doCheck = false;
                  enableLibraryProfiling = false;
                });
            });
        };
    };
}
