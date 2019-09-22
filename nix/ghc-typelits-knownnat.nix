{ mkDerivation, base, ghc, ghc-prim, ghc-tcplugins-extra
, ghc-typelits-natnormalise, stdenv, tasty, tasty-hunit, tasty-quickcheck
, template-haskell, transformers
}:
mkDerivation {
  pname = "ghc-typelits-knownnat";
  version = "0.7";
  sha256 = "00f8m3kmp572r8jr246m8r6lwzxmiqj4hml06w09l9n3lzvjwv7b";
  revision = "1";
  editedCabalFile = "1jgwa66dbhqsav7764cfcmzs3p0f3csbdjbrnbilhv1bpqyhz8sm";
  libraryHaskellDepends = [
    base ghc ghc-prim ghc-tcplugins-extra ghc-typelits-natnormalise
    template-haskell transformers
  ];
  testHaskellDepends = [
    base ghc-typelits-natnormalise tasty tasty-hunit tasty-quickcheck
  ];
  description = "Derive KnownNat constraints from other KnownNat constraints";
  license = stdenv.lib.licenses.bsd2;
  hydraPlatforms = stdenv.lib.platforms.none;
}
