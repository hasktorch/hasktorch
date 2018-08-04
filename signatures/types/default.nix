{ mkDerivation, base, stdenv }:
mkDerivation {
  pname = "hasktorch-signatures-types";
  version = "0.1.0.0";
  src = ./.;
  libraryHaskellDepends = [ base ];
  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "Backpack signature types to pair with hasktorch-raw and hasktorch-core";
  license = stdenv.lib.licenses.bsd3;
}
