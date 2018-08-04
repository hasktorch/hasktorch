{ mkDerivation, base, c2hs, hasktorch-types-th, inline-c, stdenv }:
mkDerivation {
  pname = "hasktorch-types-thc";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [ base hasktorch-types-th inline-c ];
  libraryToolDepends = [ c2hs ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Raw C-types from cutorch";
  license = stdenv.lib.licenses.bsd3;
}
