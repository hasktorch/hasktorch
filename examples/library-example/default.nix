{ ghcVersion ? "ghc865" }:
let
  hostPkgs = import <nixpkgs> { };

  srcs = {
    hasktorch = hostPkgs.fetchFromGitHub {
      owner = "hasktorch";
      repo = "hasktorch";
      rev =
        "7e017756fd9861218bf2f804d1f7eaa4d618eb01"; # use whatever revision you'd like
      sha256 =
        "1493r1bdni32j5hyw3ka3kndx3d2ydm5wmprmklfg4lj7v967n5f"; # update the sha256 value to something arbitrary when you bump the revision; nix will complain with the correct value.
    };

    all-hies = import (builtins.fetchTarball
      "https://github.com/infinisil/all-hies/tarball/master") { };
  };

  nixpkgs =
    builtins.fromJSON (builtins.readFile (srcs.hasktorch + /nix/nixpkgs.json));

in let
  hasktorchShared =
    (import (srcs.hasktorch + /nix/shared.nix) { compiler = ghcVersion; });
  # an overlay that specifies the hasktorch environment
  hasktorchOverlay = hasktorchShared.overlayShared;
  # Adds haskell-ide-engine for the specified ${ghcVersion}. (This is, of course, optional if you don't want to use hie.)  See https://github.com/infinisil/all-hies/README.md for setting up cachix to avoid building hie from scratch.
  hies = srcs.all-hies.selection {
    selector = p: { "${ghcVersion}" = p."${ghcVersion}"; };
  };

  pinnedPkgs = hostPkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    inherit (nixpkgs) rev sha256;
  };

  # an overlay that specifies haskell packages to use for ghc package set specified by ${ghcVersion}.
  haskellOverlay = pkgsNew: pkgsOld: {
    haskell = pkgsOld.haskell // {
      packages = pkgsOld.haskell.packages // {
        "${ghcVersion}" = pkgsOld.haskell.packages."${ghcVersion}".override
          (old: {
            overrides = let
              haskellPackagesExtension = hself: hsuper: {
                # other haskell package overrides can be added here.
                hasktorch = hself.hasktorch_cpu;
              };
            in pkgsNew.lib.fold pkgsNew.lib.composeExtensions
            (old.overrides or (_: _: { })) [ haskellPackagesExtension ];
          });
      };
    };
  };

  overlays = [ hasktorchOverlay haskellOverlay ];

  # overlay the two overlays onto nixpkgs
  pkgs = import pinnedPkgs {
    inherit overlays;
    config = { allowUnfree = true; };
  };

  compiler = pkgs.haskell.packages."${ghcVersion}";

# the following attributes allow:
# nix-build -A build    <- build the project
# nix-shell -A shell    <- load a nix shell with the below specified environment.
# and an `overlays` attribute in case downstream expressions need access to the overlays.
in {
  inherit overlays;

  build = compiler.callCabal2nix "library-example" ./. { };

  shell = compiler.developPackage {
    root = ./.;
    modifier = drv:
      # add any shell or build tools we'd like in this environment
      pkgs.haskell.lib.addBuildTools drv
      (with pkgs.haskellPackages; [ cabal-install ghcid hies pkgs.llvm ]);
  };
}
