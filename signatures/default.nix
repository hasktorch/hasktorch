{ mkDerivation, base, hasktorch-partial, hasktorch-raw-th
, hasktorch-raw-thc, hasktorch-signatures-types, hasktorch-types-th
, hasktorch-types-thc, stdenv
}:
mkDerivation {
  pname = "hasktorch-signatures";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    base hasktorch-partial hasktorch-signatures-types
    hasktorch-types-th hasktorch-types-thc
  ];
  executableHaskellDepends = [
    base hasktorch-raw-th hasktorch-raw-thc hasktorch-types-th
    hasktorch-types-thc
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Backpack signature files to pair with hasktorch-raw and hasktorch-core";
  license = stdenv.lib.licenses.bsd3;
}
