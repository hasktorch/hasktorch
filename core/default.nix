{ mkDerivation, backprop, base, containers, deepseq, dimensions
, ghc-typelits-natnormalise, hasktorch-indef, hasktorch-partial
, hasktorch-raw-th, hasktorch-raw-thc, hasktorch-types-th
, hasktorch-types-thc, hspec, managed, microlens
, microlens-platform, monad-loops, mtl, numeric-limits, QuickCheck
, safe-exceptions, singletons, stdenv, text, time, transformers
, typelits-witnesses
}:
mkDerivation {
  pname = "hasktorch-core";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    base containers deepseq dimensions hasktorch-indef
    hasktorch-partial hasktorch-raw-th hasktorch-raw-thc
    hasktorch-types-th hasktorch-types-thc managed microlens
    numeric-limits safe-exceptions singletons text typelits-witnesses
  ];
  testHaskellDepends = [
    backprop base dimensions ghc-typelits-natnormalise hspec
    microlens-platform monad-loops mtl QuickCheck singletons text time
    transformers
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
