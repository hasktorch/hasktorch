{
  description = "Hasktorch";

  inputs = {
    nixpkgs.follows = "haskell-nix/nixpkgs-unstable";
    haskell-nix = {
      url = "github:input-output-hk/haskell.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    utils.follows = "haskell-nix/flake-utils";
    iohkNix = {
      url = "github:input-output-hk/iohk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    libtorch-nix = {
      url = "github:stites/libtorch-nix/flakeify";
      inputs.utils.follows = "haskell-nix/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    jupyterWith = {
      url = "github:tweag/jupyterWith/35eb565c6d00f3c61ef5e74e7e41870cfa3926f7";
      flake = false;
    };

    naersk = { # should be moved into a tokenizers flake
      url = "github:nix-community/naersk";
      #flake = false;
    };

    tokenizers = {
      url = "github:hasktorch/tokenizers";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, haskell-nix, libtorch-nix, utils, iohkNix, naersk, tokenizers, ... }: with utils.lib;
    let
      inherit (nixpkgs) lib;
      inherit (lib);
      inherit (iohkNix.lib) collectExes;

      supportedSystems = ["x86_64-darwin" "x86_64-linux"];

      gitrev = self.rev or "dirty";

      profiling = true;
      cudaSupport = false;
      cudaMajorVersion = "11";

      overlays = [
        haskell-nix.overlay
        iohkNix.overlays.haskell-nix-extra

        (final: prev: {
          haskell-nix = prev.haskell-nix // {
            custom-tools = prev.haskell-nix.custom-tools // {
              haskell-language-server."1.3.0" = args':
                let
                  args = removeAttrs args' [ "version" ];
                in
                  (prev.haskell-nix.cabalProject (args // {
                    name = "haskell-language-server";
                    src = prev.fetchFromGitHub {
                      owner = "haskell";
                      repo = "haskell-language-server";
                      rev = "790afc6b920ed82e10135014e4a4ab67348d7898";
                      sha256 = "19prgrlqs66a08vq5xvfx0ms4kxcx4d2hdw565bkhd080l553nb4";
                    };
                    modules = [{
                      nonReinstallablePkgs = [
                        "rts" "ghc-heap" "ghc-prim" "integer-gmp" "integer-simple" "base"
                        "deepseq" "array" "ghc-boot-th" "pretty" "template-haskell"
                        "ghcjs-prim" "ghcjs-th"
                        "ghc-bignum" "exceptions" "stm"
                        "ghc-boot"
                        "ghc" "Cabal" "Win32" "array" "binary" "bytestring" "containers"
                        "directory" "filepath" "ghc-boot" "ghc-compact" "ghc-prim"
                        "hpc"
                        "mtl" "parsec" "process" "text" "time" "transformers"
                        "unix" "xhtml" "terminfo"
                      ];
                      # enableLibraryProfiling = true;
                      # packages.haskell-language-server.enableExecutableProfiling = true;
                      packages.haskell-language-server.components.library.ghcOptions = ["-Wall" "-Wredundant-constraints" "-Wno-name-shadowing" "-Wno-unticked-promoted-constructors" "-dynamic"];
                      packages.haskell-language-server.components.exes.haskell-language-server.ghcOptions = ["-Wall" "-Wredundant-constraints" "-Wno-name-shadowing" "-Wredundant-constraints" "-dynamic" "-rtsopts" "-with-rtsopts=-I0" "-with-rtsopts=-A128M" "-Wno-unticked-promoted-constructors"];
                      # packages.haskell-language-server.components.exes.haskell-language-server.ghcOptions = ["-Wall" "-Wredundant-constraints" "-Wno-name-shadowing" "-Wredundant-constraints" "-rtsopts" "-with-rtsopts=-I0" "-with-rtsopts=-A128M" "-with-rtsopts=-xc" "-Wno-unticked-promoted-constructors"];

                    }];
                    cabalProject = ''
                      packages:
                        ./
                        ./hie-compat
                        ./shake-bench
                        ./hls-graph
                        ./ghcide
                        ./hls-plugin-api
                        ./hls-test-utils
                      --  ./plugins/hls-tactics-plugin
                      --  ./plugins/hls-brittany-plugin
                      --  ./plugins/hls-stylish-haskell-plugin
                      --  ./plugins/hls-fourmolu-plugin
                        ./plugins/hls-class-plugin
                        ./plugins/hls-eval-plugin
                        ./plugins/hls-explicit-imports-plugin
                        ./plugins/hls-refine-imports-plugin
                        ./plugins/hls-hlint-plugin
                        ./plugins/hls-retrie-plugin
                        ./plugins/hls-haddock-comments-plugin
                      --  ./plugins/hls-splice-plugin
                        ./plugins/hls-floskell-plugin
                        ./plugins/hls-pragmas-plugin
                        ./plugins/hls-module-name-plugin
                        ./plugins/hls-ormolu-plugin
                        ./plugins/hls-call-hierarchy-plugin

                      constraints:
                        haskell-language-server -brittany -class -fourmolu -splice -stylishhaskell -tactic -refineImports

                      source-repository-package
                        type: git
                        location: https://github.com/jwaldmann/blaze-textual.git
                        tag: d8ee6cf80e27f9619d621c936bb4bda4b99a183f
                        --sha256: 0k1xv17f4dk67d6ina3hrljvj009cink4qb9yac1cz5qzv6lhiqb

                      source-repository-package
                        type: git
                        location: https://github.com/mithrandi/czipwith.git
                        tag: b6245884ae83e00dd2b5261762549b37390179f8
                        --sha256: 0hapj3n8vnk2xx1vqn6v6g10kzn0cjgcfa8pnnng6kzi58dsir6s

                      source-repository-package
                        type: git
                        location: https://github.com/jneira/hie-bios/
                        tag: 9b1445ab5efcabfad54043fc9b8e50e9d8c5bbf3
                        --sha256: 0jfm7shlkb8vg2srprabvsnhmr77bvp59z771cl81i28gjvppjzi

                      source-repository-package
                        type: git
                        location: https://github.com/hsyl20/ghc-api-compat
                        tag: 8fee87eac97a538dbe81ff1ab18cff10f2f9fa15
                        --sha256: sha256-byehvdxQxhNk5ZQUXeFHjAZpAze4Ct9261ro4c5acZk=

                      source-repository-package
                        type: git
                        location: https://github.com/anka-213/th-extras
                        tag: 57a97b4df128eb7b360e8ab9c5759392de8d1659
                        --sha256: 1yg0ikw63kmgp35kwhdi63sbk4f6g7bdj5app3z442xz0zb5mn22

                      source-repository-package
                        type: git
                        location: https://github.com/anka-213/dependent-sum
                        tag: 8cf4c7fbc3bfa2be475a17bb7c94a1e1e9a830b5
                        subdir: dependent-sum-template
                        --sha256: 0x3lgd1ckd1666infydx9iijdvllw4kikb8k9fx90kczmc3m7p2s

                      source-repository-package
                        type: git
                        location: https://github.com/HeinrichApfelmus/operational
                        tag: 16e19aaf34e286f3d27b3988c61040823ec66537
                        --sha256: 1831g81pnx8sn0w1j9srs37bmai2wv521dvmjqjdy21a8xqairiz

                      allow-newer:
                        assoc:base,
                        cryptohash-md5:base,
                        cryptohash-sha1:base,
                        constraints-extras:template-haskell,
                        data-tree-print:base,
                        deepseq:base,
                        dependent-sum:some,
                        dependent-sum:constraints,
                        diagrams-postscript:base,
                        diagrams-postscript:lens,
                        diagrams-postscript:diagrams-core,
                        diagrams-postscript:monoid-extras,
                        diagrams:diagrams-core,
                        Chart-diagrams:diagrams-core,
                        SVGFonts:diagrams-core,
                        dual-tree:base,
                        entropy:Cabal,
                        force-layout:base,
                        force-layout:lens,
                        floskell:ghc-prim,
                        floskell:base,
                        hashable:base,
                        hslogger:base,
                        monoid-extras:base,
                        newtype-generics:base,
                        parallel:base,
                        regex-base:base,
                        regex-tdfa:base,
                        statestack:base,
                        svg-builder:base,
                        these:base,
                        time-compat:base
                    '';
                  })).haskell-language-server.components.exes.haskell-language-server;
            };
          };
        })

        (if !cudaSupport then libtorch-nix.overlays.cpu
         else if (cudaMajorVersion == "10") then libtorch-nix.overlays.cudatoolkit_10_2
         else libtorch-nix.overlays.cudatoolkit_11_1)

        (final: prev: {
          inherit gitrev;
          commonLib = lib
            // iohkNix.lib;
        })

        (final: prev: {
          naersk = naersk.lib."${prev.system}";
        })
        (import "${tokenizers}/nix/pkgs.nix")
        (final: prev: {
          tokenizers_haskell = prev.tokenizersPackages.tokenizers-haskell;
        })

        (final: prev: {
          hasktorchProject = import ./nix/haskell.nix (rec {
            pkgs = prev;
            compiler-nix-name = "ghc901";
            inherit (prev) lib;
            inherit profiling;
            inherit cudaSupport;
          });
        })
      ];

    in eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs { inherit system overlays; };

        legacyPkgs = haskell-nix.legacyPackages.${system}.appendOverlays overlays;

        inherit (pkgs.commonLib) eachEnv environments;

        devShell =  import ./shell.nix {
          inherit pkgs;
          inherit cudaSupport;
          inherit cudaMajorVersion;
        };

        flake = pkgs.hasktorchProject.flake {};

        checks = collectChecks flake.packages;

        exes = collectExes flake.packages;

      in lib.recursiveUpdate flake {
        inherit environments checks legacyPkgs;

        defaultPackage = flake.packages."hasktorch:lib:hasktorch";

        inherit devShell;
      }
    );
}
