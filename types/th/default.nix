{ mkDerivation, base, c2hs, inline-c, stdenv }:
mkDerivation {
  pname = "hasktorch-types-th";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [ base inline-c ];
  libraryToolDepends = [ c2hs ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Raw C-types from torch";
  license = stdenv.lib.licenses.bsd3;
}
