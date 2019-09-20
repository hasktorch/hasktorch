{ compiler ? "ghc865" }:

let
  libtorch_src = pkgsOld:
    let src = pkgsOld.fetchFromGitHub {
          owner  = "junjihashimoto";
          repo   = "pytorch-world";
          rev    = "3fab6c6ec237b63ed8c61a90efb5c3bb11686037";
          sha256 = "1ghjfz9fc7m8877bqkk37ikssc9qjf4bqb75aiyhrx7gxp4km3dh";
    };
    in (pkgsOld.callPackage "${src}/libtorch/release.nix" { });
  overlayShared = pkgsNew: pkgsOld: {
    libtorch = (libtorch_src pkgsOld).libtorch_cpu;
    haskell = pkgsOld.haskell // {
      packages = pkgsOld.haskell.packages // {
        "${compiler}" = pkgsOld.haskell.packages."${compiler}".override (old: {
            overrides =
              let
                appendConfigureFlag = pkgsNew.haskell.lib.appendConfigureFlag;
                dontCheck = pkgsNew.haskell.lib.dontCheck;
                failOnAllWarnings = pkgsNew.haskell.lib.failOnAllWarnings;
                overrideCabal = pkgsNew.haskell.lib.overrideCabal;
                optionalString = pkgsNew.stdenv.lib.optionalString;
                isDarwin = pkgsNew.stdenv.isDarwin;

                extension =
                  haskellPackagesNew: haskellPackagesOld: {
                    hasktorch =
                      overrideCabal
                        (haskellPackagesNew.callCabal2nix
                          "hasktorch"
                          ../hasktorch
                          { }
                        )
                        (old: {
                              preConfigure = (old.preConfigure or "") + optionalString (!isDarwin) ''
                                export LD_PRELOAD=${pkgs.mkl}/lib/libmkl_rt.so
                              '';
                            }
                        );
                    hasktorch-codegen =
                      # failOnAllWarnings
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
                        (overrideCabal
                          (haskellPackagesNew.callCabal2nix
                            "libtorch-ffi"
                            ../libtorch-ffi
                             { c10 = pkgsNew.libtorch; torch = pkgsNew.libtorch; }
                          )
                          (old: {
                              preConfigure = (old.preConfigure or "") + optionalString isDarwin ''
                                sed -i -e 's/-optc-std=c++11 -optc-xc++/-optc-xc++/g' ../libtorch-ffi/libtorch-ffi.cabal;
                              '';
                            }
                          )
                        )
                        "--extra-include-dirs=${pkgsNew.libtorch}/include/torch/csrc/api/include";
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

    shell-hasktorch = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch).env.overrideAttrs
      (oldAttrs:
        oldAttrs // {
          shellHook = ''
            export LD_PRELOAD=${pkgs.mkl}/lib/libmkl_rt.so
          '';
        }
      );
    shell-hasktorch-codegen = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-codegen).env;
    shell-hasktorch-examples = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-examples).env;
    shell-libtorch-ffi = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi).env.overrideAttrs
      (oldAttrs:
        oldAttrs // {
          shellHook = ''
            export CPATH=${pkgs.libtorch}/include/torch/csrc/api/include
          '';
        }
      );
    shell-inline-c = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c).env;
    shell-inline-c-cpp = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c-cpp).env;
  }

