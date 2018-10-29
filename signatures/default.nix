{ mkDerivation, base, hasktorch-partial, hasktorch-ffi-th
, hasktorch-ffi-thc, hasktorch-signatures-types, hasktorch-types-th
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
    base hasktorch-ffi-th hasktorch-ffi-thc hasktorch-types-th
    hasktorch-types-thc
  ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Backpack signature files to pair with hasktorch-ffi and hasktorch-core";
  license = stdenv.lib.licenses.bsd3;
}
