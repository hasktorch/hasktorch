{ mkDerivation, base, containers, ghc, ghc-tcplugins-extra
, integer-gmp, stdenv, syb, tasty, tasty-hunit, template-haskell
, transformers
}:
mkDerivation {
  pname = "ghc-typelits-natnormalise";
  version = "0.7.2";
  sha256 = "a24ccd647cc8a12bf35ceacf58773d35f1e005498ee37bf63c2ade89b77062c2";
  libraryHaskellDepends = [
    base containers ghc ghc-tcplugins-extra integer-gmp syb
    transformers
  ];
  testHaskellDepends = [ base tasty tasty-hunit template-haskell ];
  homepage = "http://www.clash-lang.org/";
  description = "GHC typechecker plugin for types of kind GHC.TypeLits.Nat";
  license = stdenv.lib.licenses.bsd2;
}
