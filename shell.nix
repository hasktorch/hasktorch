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

    # packages = ps: with ps; [ ];

    tools = {
      cabal = "latest";
      # haskell-language-server = "latest";
    };

    # buildInputs = with haskellPackages; [ ];

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
