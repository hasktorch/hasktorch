{ config ? {}
, sourcesOverride ? {}
, cudaSupport
, cudaMajorVersion
, pkgs
}:

with pkgs;

let

  shell = pkgs.hasktorchProject.shellFor {
    name = "hasktorch-dev-shell";

    # If shellFor local packages selection is wrong,
    # then list all local packages then include source-repository-package that cabal complains about:
    #packages = ps: with ps; [ ];

    tools = {
      cabal = "latest";
      haskell-language-server = "latest";
    };

    # These programs will be available inside the nix-shell.
    buildInputs = with haskellPackages; [ ];

    exactDeps = true;

    shellHook =
      let
        cpath = ''
          export CPATH=${torch}/include/torch/csrc/api/include
        '';
        nproc = ''
          case "$(uname)" in
            "Linux")
                ${utillinux}/bin/taskset -pc 0-1000 $$
            ;;
          esac
        '';
        libraryPath = lib.optionalString cudaSupport ''
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
        '';
        tokenizersLibraryPath = ''
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${tokenizers_haskell}/lib"
        '';
      in
        cpath + nproc + libraryPath + tokenizersLibraryPath;

    withHoogle = true;
  };

in

  shell
