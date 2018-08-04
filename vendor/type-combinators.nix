{ mkDerivation, base, stdenv }:
mkDerivation {
  pname = "type-combinators";
  version = "0.2.4.3";
  sha256 = "1xip4gav1fn3ip62mrlbr7p6i1254fa1q542cmp6ffzm55lwn30z";
  libraryHaskellDepends = [ base ];
  homepage = "https://github.com/kylcarte/type-combinators";
  description = "A collection of data types for type-level programming";
  license = stdenv.lib.licenses.bsd3;
}
