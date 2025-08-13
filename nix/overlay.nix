final: prev: let
  inherit (final) lib libtorch-bin;
  inherit
    (final.haskell.lib)
    overrideSrc
    ;
  inherit
    (final.haskell.lib.compose)
    appendConfigureFlag
    doJailbreak
    dontCheck
    ;
  torch = libtorch-bin;
  c10 = libtorch-bin;
  torch_cpu = libtorch-bin;
  ghcName = "ghc984";
in {
  haskell =
    prev.haskell
    // {
      packages =
        prev.haskell.packages
        // {
          ${ghcName} =
            prev.haskell.packages.${ghcName}.extend
            (hfinal: hprev: {
              # Hasktorch Packages
              codegen = hfinal.callCabal2nix "codegen" ../codegen {};
              # hasktorch-gradually-typed =
              #   lib.pipe
              #   (hfinal.callCabal2nix "hasktorch-gradually-typed" ../experimental/gradually-typed {})
              #   [
              #     dontCheck
              #     #  disableLibraryProfiling
              #   ];
              hasktorch = hfinal.callCabal2nix "hasktorch" ../hasktorch {};
              libtorch-ffi-helper = hfinal.callCabal2nix "libtorch-ffi-helper" ../libtorch-ffi-helper {};
              libtorch-ffi =
                lib.pipe
                (hfinal.callCabal2nix "libtorch-ffi" ../libtorch-ffi {inherit torch c10 torch_cpu;})
                [
                  (appendConfigureFlag
                    "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include")
                ];

              # Hasktorch Forks
              # WARNING: Does not build with GHC 9.8
              # typelevel-rewrite-rules =
              #   doJailbreak((overrideSrc hprev.typelevel-rewrite-rules {
              #     src = prev.fetchFromGitHub {
              #       owner = "hasktorch";
              #       repo = "typelevel-rewrite-rules";
              #       rev = "1f181c3073df201cec45e121f07610b0bfbb6ecd";
              #       sha256 = "sha256-CbauA2leHYtdCT0tiDeRCNfJddc/5x9sPz+stmHVR5Q=";
              #     };
              #   })
              #   .override {
              #     term-rewriting = overrideSrc hprev.term-rewriting {
              #       src = prev.fetchFromGitHub {
              #         owner = "hasktorch";
              #         repo = "term-rewriting";
              #         rev = "54221f58b28c9f36db9bac437231e6142c8cca3a";
              #         sha256 = "sha256-cDthJ+XJ7J8l0SFpPRnvFt2yC4ufD6efz5GES5xMtzQ=";
              #       };
              #     };
              #   });
              # Applies the same changes available in hasktorch/type-errors-pretty
              # type-errors-pretty = lib.pipe hprev.type-errors-pretty [doJailbreak dontCheck];

              # Dependency Fixes
              # indexed-extras = overrideSrc hprev.indexed-extras {
              #   src = prev.fetchFromGitHub {
              #     owner = "reinerp";
              #     repo = "indexed-extras";
              #     rev = "7a0c4e918578e7620a46d4f0546fbdaec933ede0";
              #     sha256 = "sha256-SS6yZEKOZ5aRgPW7xMtojNDr0TjZ+3QHCgf/o9umG84=";
              #   };
              # };
              singletons-base = dontCheck hprev.singletons-base;
            });
        };
    };
}
