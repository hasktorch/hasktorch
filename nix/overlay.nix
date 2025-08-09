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
  libtorch_2_5_0_cpu = final.fetchzip {
    name = "libtorch-2.5.0-cpu.zip";
    url  = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip";
    hash = "sha256-gUzPhc4Z8rTPhIm89pPoLP0Ww17ono+/xgMW46E/Tro=";
    stripRoot = false;
  };
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
              libtorch-2_5_0-cpu = libtorch_2_5_0_cpu;
              codegen = (hfinal.callCabal2nix "codegen" ../codegen {}).overrideAttrs (old: {
                  preConfigure = (old.preConfigure or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                  preBuild = (old.preBuild or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                  preInstall = (old.preInstall or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                });
              hasktorch-gradually-typed =
                lib.pipe
                (hfinal.callCabal2nix "hasktorch-gradually-typed" ../experimental/gradually-typed {})
                [
                  (drv: drv.overrideAttrs (old: {
                    preConfigure = (old.preConfigure or "") + ''
                      export HOME="$TMPDIR"
                      export XDG_CACHE_HOME="$TMPDIR"
                      export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                      export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                      export LIBTORCH_SKIP_DOWNLOAD=1
                      ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                      export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                      export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                    '';
                    preBuild = (old.preBuild or "") + ''
                      export HOME="$TMPDIR"
                      export XDG_CACHE_HOME="$TMPDIR"
                      export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                      export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                      export LIBTORCH_SKIP_DOWNLOAD=1
                      ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                      export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                      export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                    '';
                    preInstall = (old.preInstall or "") + ''
                      export HOME="$TMPDIR"
                      export XDG_CACHE_HOME="$TMPDIR"
                      export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                      export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                      export LIBTORCH_SKIP_DOWNLOAD=1
                      ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                      export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                      export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                    '';
                  }))
                  dontCheck
                  #  disableLibraryProfiling
                ];
              hasktorch =
                lib.pipe
                  (hfinal.callCabal2nix "hasktorch" ../hasktorch {})
                  [
                    (drv: drv.overrideAttrs (old: {
                      preConfigure = (old.preConfigure or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                      preBuild = (old.preBuild or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                      preInstall = (old.preInstall or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                    }))
                    dontCheck
                    # disableLibraryProfiling
                  ];

              libtorch-ffi-helper =
                (hfinal.callCabal2nix "libtorch-ffi-helper" ../libtorch-ffi-helper {}).overrideAttrs (old: {
                  preConfigure = (old.preConfigure or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                  preBuild = (old.preBuild or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                  preInstall = (old.preInstall or "") + ''
                    export HOME="$TMPDIR"
                    export XDG_CACHE_HOME="$TMPDIR"
                    export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                    export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                    export LIBTORCH_SKIP_DOWNLOAD=1
                    ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                    export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                    export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                  '';
                });

              libtorch-ffi =
                lib.pipe
                  (hfinal.callCabal2nix "libtorch-ffi" ../libtorch-ffi {
                    inherit torch c10 torch_cpu;
                  })
                  [
                    (drv: drv.overrideAttrs (old: {
                      preConfigure = (old.preConfigure or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                      preBuild = (old.preBuild or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                      preInstall = (old.preInstall or "") + ''
                        export HOME="$TMPDIR"
                        export XDG_CACHE_HOME="$TMPDIR"
                        export HASKTORCH_LIB_PATH="$XDG_CACHE_HOME/libtorch/lib:$XDG_CACHE_HOME/mklml/lib/:$XDG_CACHE_HOME/libtokenizers/lib"
                        export LD_LIBRARY_PATH=$HASKTORCH_LIB_PATH:$LD_LIBRARY_PATH
                        export LIBTORCH_SKIP_DOWNLOAD=1
                        ln -sfn ${libtorch_2_5_0_cpu}/libtorch "$XDG_CACHE_HOME/libtorch"
                        export LIBTORCH_HOME="$XDG_CACHE_HOME/libtorch"
                        export CMAKE_PREFIX_PATH="$LIBTORCH_HOME"
                      '';
                    }))
                    (appendConfigureFlag
                      "--extra-include-dirs=${lib.getDev torch}/include/torch/csrc/api/include")
                  ];

              # Hasktorch Forks
              # WARNING: Does not build with GHC 9.8
              typelevel-rewrite-rules =
                doJailbreak((overrideSrc hprev.typelevel-rewrite-rules {
                  src = prev.fetchFromGitHub {
                    owner = "hasktorch";
                    repo = "typelevel-rewrite-rules";
                    rev = "1f181c3073df201cec45e121f07610b0bfbb6ecd";
                    sha256 = "sha256-CbauA2leHYtdCT0tiDeRCNfJddc/5x9sPz+stmHVR5Q=";
                  };
                })
                .override {
                  term-rewriting = overrideSrc hprev.term-rewriting {
                    src = prev.fetchFromGitHub {
                      owner = "hasktorch";
                      repo = "term-rewriting";
                      rev = "54221f58b28c9f36db9bac437231e6142c8cca3a";
                      sha256 = "sha256-cDthJ+XJ7J8l0SFpPRnvFt2yC4ufD6efz5GES5xMtzQ=";
                    };
                  };
                });
              # Applies the same changes available in hasktorch/type-errors-pretty
              type-errors-pretty = lib.pipe hprev.type-errors-pretty [doJailbreak dontCheck];

              # Dependency Fixes
              indexed-extras = overrideSrc hprev.indexed-extras {
                src = prev.fetchFromGitHub {
                  owner = "reinerp";
                  repo = "indexed-extras";
                  rev = "7a0c4e918578e7620a46d4f0546fbdaec933ede0";
                  sha256 = "sha256-SS6yZEKOZ5aRgPW7xMtojNDr0TjZ+3QHCgf/o9umG84=";
                };
              };
              singletons-base = dontCheck hprev.singletons-base;
            });
        };
    };
}
