{ mkDerivation, base, containers, directory, hashable, hspec
, hspec-discover, megaparsec, optparse-applicative, pretty-show
, QuickCheck, stdenv, text, unordered-containers
}:
mkDerivation {
  pname = "hasktorch-codegen";
  version = "0.0.1.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    base containers directory hashable megaparsec pretty-show text
    unordered-containers
  ];
  executableHaskellDepends = [
    base optparse-applicative pretty-show
  ];
  testHaskellDepends = [
    base containers hspec hspec-discover megaparsec pretty-show
    QuickCheck text
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
