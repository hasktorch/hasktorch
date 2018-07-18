{ mkDerivation, base, hasktorch-signatures-types
, hasktorch-types-th, inline-c, stdenv, text
}:
mkDerivation {
  pname = "hasktorch-partial";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    base hasktorch-signatures-types hasktorch-types-th inline-c text
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
