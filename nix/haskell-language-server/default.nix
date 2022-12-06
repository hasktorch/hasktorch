{ pkgs, ... }: {
  haskell-language-server."1.8.0.0" = args':
    let
      args = removeAttrs args' [ "version" ];
    in
      (pkgs.haskell-nix.cabalProject (args // {
        name = "haskell-language-server";
        src = pkgs.fetchFromGitHub {
          owner = "haskell";
          repo = "haskell-language-server";
          rev = "68d353f1ed42f3643bdb6043244b491030cc3e99";
          sha256 = "1nkvx4psahl1p2scm91sy5dc4bck5cslspz83w2r0z30651k1rhw";
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
