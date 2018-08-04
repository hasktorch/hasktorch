{ mkDerivation, ATen, base, c2hs, hasktorch-raw-tests
, hasktorch-types-th, hspec, inline-c, QuickCheck, stdenv, text
}:
mkDerivation {
  pname = "hasktorch-raw-th";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [ base hasktorch-types-th inline-c text ];
  librarySystemDepends = [ ATen ];
  libraryToolDepends = [ c2hs ];
  testHaskellDepends = [
    base hasktorch-raw-tests hasktorch-types-th hspec QuickCheck text
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
