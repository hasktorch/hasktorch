{ pkgs, ... }: {
  haskell-language-server."1.5.0.0" = args':
    let
      args = removeAttrs args' [ "version" ];
    in
      (pkgs.haskell-nix.cabalProject (args // {
        name = "haskell-language-server";
        src = pkgs.fetchFromGitHub {
          owner = "haskell";
          repo = "haskell-language-server";
          rev = "9233be8ec7666555462560fd2637586cadad1898";
          sha256 = "1w87gwnvkb17l7rzyx7qni4hspjfk6qc6yxlr8mxi384d3dpsap0";
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
        cabalProject = builtins.readFile ./cabal.project;
      })).haskell-language-server.components.exes.haskell-language-server;
}
