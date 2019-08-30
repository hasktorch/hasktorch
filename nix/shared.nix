{ compiler ? "ghc865" }:

let
  overlayShared = pkgsNew: pkgsOld: {
    libtorch-cuda =
      let src = pkgsOld.fetchFromGitHub {
          owner  = "stites";
          repo   = "pytorch-world";
          rev    = "4c4bd205e47477c723209f3677bddceabb614237";
          sha256 = "1zqhm0874gfs1qp3glc1i1dswlrksby1wf9dr4pclkabs0smbqxc";
        };
      in
      (import "${src}/release.nix" { }).libtorch-cuda;
    haskell = pkgsOld.haskell // {
      packages = pkgsOld.haskell.packages // {
        "${compiler}" = pkgsOld.haskell.packages."${compiler}".override (old: {
            overrides =
              let
                failOnAllWarnings = pkgsOld.haskell.lib.failOnAllWarnings;
                overrideExtraLibraries = drv: xs: pkgsOld.haskell.lib.overrideCabal drv (drv: { extraLibraries = xs; });

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
                    libtorch-ffi =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "libtorch-ffi"
                          ../libtorch-ffi
                          { }
                        );
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
    config = {};
    overlays = [ overlayShared ];
  };

in
  rec {
    inherit (pkgs.haskell.packages."${compiler}")
      hasktorch
      hasktorch-codegen
      libtorch-ffi
      inline-c
      inline-c-cpp
    ;

    shell-hasktorch = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch).env;
    shell-hasktorch-codegen = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-codegen).env;
    shell-libtorch-ffi = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi).env;
    shell-inline-c = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c).env;
    shell-inline-c-cpp = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c-cpp).env;
  }

