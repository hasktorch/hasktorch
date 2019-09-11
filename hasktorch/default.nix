{ mkDerivation, base, finite-typelits, ghc-typelits-knownnat, hspec
, hspec-discover, libtorch-ffi, mtl, QuickCheck, reflection
, safe-exceptions, stdenv
}:
mkDerivation {
  pname = "hasktorch";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [
    base finite-typelits ghc-typelits-knownnat libtorch-ffi mtl
    reflection safe-exceptions
  ];
  testHaskellDepends = [
    base hspec hspec-discover mtl QuickCheck reflection safe-exceptions
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "initial implementation for hasktorch based on libtorch";
  license = stdenv.lib.licenses.bsd3;
}
