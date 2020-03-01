_: pkgs:

let
  ihaskellSrc = pkgs.fetchFromGitHub {
    owner = "gibiansky";
    repo = "IHaskell";
    rev = "2318ee2a90cfc98390651657aec434586b963235";
    sha256 = "0svjzs81i77s710cfb7pxkfdi979mhjazpc2l9k9ha752spz04cj";
  };

  monadBayesSrc = pkgs.fetchFromGitHub {
    owner = "adscib";
    repo = "monad-bayes";
    rev = "fb87bf039bab35dcc82de8ccf8963a7a576af355";
    sha256 = "0jz7lswdzxzn5zzwypdawdj7j0y20aakmqggv9pw4sknajdqqqyf";
  };

  hVegaSrc = pkgs.fetchFromGitHub {
    owner = "DougBurke";
    repo = "hvega";
    rev = "hvega-0.4.0.0";
    sha256 = "1pg655a36nsz7h2l1sbyk4zzzjjw4dlah8794bc0flpigr7iik13";
  };

  overrides = self: hspkgs:
    let
      callDisplayPackage = name:
        hspkgs.callCabal2nix
          "ihaskell-${name}"
          "${ihaskellSrc}/ihaskell-display/ihaskell-${name}"
          {};
      dontCheck = pkgs.haskell.lib.dontCheck;
      dontHaddock = pkgs.haskell.lib.dontHaddock;
    in
    {
      monad-bayes = hspkgs.callCabal2nix "monad-bayes" "${monadBayesSrc}" {};
      hvega = hspkgs.callCabal2nix "hvega" "${hVegaSrc}/hvega" {};
      ihaskell-hvega = hspkgs.callCabal2nix "ihaskell-hvega" "${hVegaSrc}/ihaskell-hvega" {};
      ihaskell = pkgs.haskell.lib.overrideCabal
        (hspkgs.callCabal2nix "ihaskell" ihaskellSrc {})
        (_drv: {
          preCheck = ''
            export HOME=$(${pkgs.pkgs.coreutils}/bin/mktemp -d)
            export PATH=$PWD/dist/build/ihaskell:$PATH
            export GHC_PACKAGE_PATH=$PWD/dist/package.conf.inplace/:$GHC_PACKAGE_PATH
          '';
          configureFlags = (_drv.configureFlags or []) ++ [
            # otherwise the tests are agonisingly slow and the kernel times out
            "--enable-executable-dynamic"
          ];
          doHaddock = false;
         });
      ghc-parser = hspkgs.callCabal2nix "ghc-parser" "${ihaskellSrc}/ghc-parser" {};
      ipython-kernel = hspkgs.callCabal2nix "ipython-kernel" "${ihaskellSrc}/ipython-kernel" {};
      ihaskell-aeson = callDisplayPackage "aeson";
      ihaskell-blaze = callDisplayPackage "blaze";
      ihaskell-charts = callDisplayPackage "charts";
      ihaskell-diagrams = callDisplayPackage "diagrams";
      ihaskell-gnuplot = callDisplayPackage "gnuplot";
      ihaskell-graphviz = callDisplayPackage "graphviz";
      ihaskell-hatex = callDisplayPackage "hatex";
      ihaskell-juicypixels = callDisplayPackage "juicypixels";
      ihaskell-magic = callDisplayPackage "magic";
      ihaskell-plot = callDisplayPackage "plot";
      ihaskell-rlangqq = callDisplayPackage "rlangqq";
      ihaskell-static-canvas = callDisplayPackage "static-canvas";
      ihaskell-widgets = callDisplayPackage "widgets";

      # Marked as broken in this version of Nixpkgs.
      chell = hspkgs.callHackage "chell" "0.4.0.2" {};
      patience = hspkgs.callHackage "patience" "0.1.1" {};

      # Version compatible with ghc-lib-parser.
      hlint = hspkgs.callHackage "hlint" "2.2.1" {};

      # Tests not passing.
      Diff = dontCheck hspkgs.Diff;
      zeromq4-haskell = dontCheck hspkgs.zeromq4-haskell;
      funflow = dontCheck hspkgs.funflow;

      # Haddocks not building.
      ghc-lib-parser = dontHaddock hspkgs.ghc-lib-parser;

      # Missing dependency.
      aeson = pkgs.haskell.lib.addBuildDepends hspkgs.aeson [ self.contravariant ];


    };
in

{
  haskell = pkgs.haskell // {
    packages = pkgs.haskell.packages // {
      "ghc865" = pkgs.haskell.packages.ghc865.override (old: {
          overrides =
              pkgs.lib.composeExtensions
                (old.overrides or (_: _: {}))
                overrides;}
              );
            };
          };
}
