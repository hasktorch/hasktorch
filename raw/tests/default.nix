{ mkDerivation, base, hasktorch-types-th, hspec, QuickCheck, stdenv
, text
}:
mkDerivation {
  pname = "hasktorch-raw-tests";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    base hasktorch-types-th hspec QuickCheck text
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Testing library for raw TH and THC bindings";
  license = stdenv.lib.licenses.bsd3;
}
