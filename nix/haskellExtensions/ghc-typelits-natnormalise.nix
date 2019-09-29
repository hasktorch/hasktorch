{ mkDerivation, base, containers, ghc, ghc-tcplugins-extra
, integer-gmp, stdenv, tasty, tasty-hunit, template-haskell, transformers
}:
mkDerivation {
  pname = "ghc-typelits-natnormalise";
  version = "0.7";
  sha256 = "1rfw67hhhka3ga8v3224ain7jvdc390glz5cr7vvxm1yqs1wgwx4";
  libraryHaskellDepends = [
    base containers ghc ghc-tcplugins-extra integer-gmp transformers
  ];
  testHaskellDepends = [ base tasty tasty-hunit template-haskell ];
  description = "GHC typechecker plugin for types of kind GHC.TypeLits.Nat";
  license = stdenv.lib.licenses.bsd2;
  hydraPlatforms = stdenv.lib.platforms.none;
}
