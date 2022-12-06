{ cudaSupport
, cudaMajorVersion
, hasktorchProject
, torch
, utillinux
, lib
, tokenizers-haskell
#, preCommitShellHook
}:

hasktorchProject.shellFor {
  name = "hasktorch-dev-shell";
  exactDeps = true;
  withHoogle = true;

  tools = {
    cabal = "latest";
    haskell-language-server = "latest";
  };

  # packages = ps: with ps; [ ];
  # buildInputs = with haskellPackages; [ ];

  shellHook = lib.strings.concatStringsSep "\n" [
    # set dynamic libraries for cuda support
    (lib.optionalString cudaSupport "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:/run/opengl-driver/lib\"")

    # put tokenizers on path
    "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:${tokenizers-haskell}/lib\""

    # make sure number of processes is set
    ''
    case "$(uname)" in
      "Linux")
          ${utillinux}/bin/taskset -pc 0-1000 $$
      ;;
    esac
    ''
    
#    preCommitShellHook
  ];
}
