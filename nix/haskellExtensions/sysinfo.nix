{ mkDerivation, base, hspec, hspec-expectations, stdenv }:
mkDerivation {
  pname = "sysinfo";
  version = "0.1.1";
  sha256 = "0afa9nv1sf1c4w2d9ysm0ass4a48na1mb3x9ri3nb5c6s7r41ns6";
  libraryHaskellDepends = [ base ];
  testHaskellDepends = [ base hspec hspec-expectations ];
  description = "Haskell Interface for getting overall system statistics";
  license = stdenv.lib.licenses.bsd3;
  doCheck = false;
}