{ mkDerivation, backprop, base, containers, deepseq, dimensions
, ghc-typelits-natnormalise, hasktorch-raw-th, hasktorch-signatures
, hasktorch-types-th, hasktorch-types-thc, managed, microlens, mtl
, numeric-limits, safe-exceptions, singletons, stdenv, text
, transformers, typelits-witnesses
}:
mkDerivation {
  pname = "hasktorch-indef";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    backprop base containers deepseq dimensions
    ghc-typelits-natnormalise hasktorch-raw-th hasktorch-signatures
    hasktorch-types-th hasktorch-types-thc managed microlens mtl
    numeric-limits safe-exceptions singletons text transformers
    typelits-witnesses
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
