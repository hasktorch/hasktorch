{ compiler ? "ghc865" }:

let
  overlayShared = pkgsNew: pkgsOld: {
    libtorch =
      let src = pkgsOld.fetchFromGitHub {
          owner  = "stites";
          repo   = "pytorch-world";
          rev    = "7178a0e61edee422266dae5d984034c3705fce91";
          sha256 = "0hzl3l3s1nalfiwbh30qswi31irz6bgvrr17brg04hzdspq7rygn";
        };
      in
      (pkgsOld.python37Packages.callPackage "${src}/pytorch/default.nix" { mklSupport = true; }).dev;
    #pytorch = pkgsOld.python3Packages.pytorchWithoutCuda.override {
    #  mklSupport = true;
    #};
    haskell = pkgsOld.haskell // {
      packages = pkgsOld.haskell.packages // {
        "${compiler}" = pkgsOld.haskell.packages."${compiler}".override (old: {
            overrides =
              let
                appendConfigureFlag = pkgsNew.haskell.lib.appendConfigureFlag;
                dontCheck = pkgsNew.haskell.lib.dontCheck;
                failOnAllWarnings = pkgsNew.haskell.lib.failOnAllWarnings;
                overrideExtraLibraries = drv: xs: pkgsNew.haskell.lib.overrideCabal drv (drv: { extraLibraries = xs; });

                extension =
                  haskellPackagesNew: haskellPackagesOld: {
                    hasktorch =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "hasktorch"
                          ../hasktorch
                          { }
                        );
                    hasktorch-codegen =
                      failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "codegen"
                          ../codegen
                          { }
                        );
                    hasktorch-examples =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "examples"
                          ../examples
                          { }
                        );
                    libtorch-ffi =
                      appendConfigureFlag
                        (haskellPackagesNew.callCabal2nix
                          "libtorch-ffi"
                          ../libtorch-ffi
                          { c10 = pkgsNew.libtorch; iomp5 = pkgsNew.mkl; torch = pkgsNew.libtorch; }
                        ) "--extra-include-dirs=${pkgsNew.libtorch}/include/torch/csrc/api/include";
                    inline-c =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "inline-c"
                          ../inline-c/inline-c
                          { }
                        );
                    inline-c-cpp =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "inline-c-cpp"
                          ../inline-c/inline-c-cpp
                          { }
                        );
                  };

              in
                pkgsNew.lib.fold
                  pkgsNew.lib.composeExtensions
                  (old.overrides or (_: _: {}))
                  [ (pkgsNew.haskell.lib.packagesFromDirectory { directory = ./.; })

                    extension
                  ];
          }
        );
      };
    };
  };

  bootstrap = import <nixpkgs> { };

  nixpkgs = builtins.fromJSON (builtins.readFile ./nixpkgs.json);

  src = bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo  = "nixpkgs";
    inherit (nixpkgs) rev sha256;
  };

  pkgs = import src {
    config = { allowUnsupportedSystem = true; allowUnfree = true; };
    overlays = [ overlayShared ];
  };

in
  rec {
    inherit (pkgs.haskell.packages."${compiler}")
      hasktorch
      hasktorch-codegen
      hasktorch-examples
      libtorch-ffi
      inline-c
      inline-c-cpp
    ;

    shell-hasktorch = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch).env.overrideAttrs (oldAttrs: oldAttrs // {
      shellHook = ''
        export LD_PRELOAD=${pkgs.mkl}/lib/libmkl_core.so:${pkgs.mkl}/lib/libmkl_sequential.so
      '';
    });
    shell-hasktorch-codegen = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-codegen).env;
    shell-hasktorch-examples = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-examples).env;
    shell-libtorch-ffi = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi).env;
    shell-inline-c = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c).env;
    shell-inline-c-cpp = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c-cpp).env;
  }

