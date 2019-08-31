{ mkDerivation, ansi-wl-pprint, base, bytestring, containers
, hashable, hspec, mtl, parsec, parsers, QuickCheck
, raw-strings-qq, regex-posix, stdenv, template-haskell
, transformers, unordered-containers, vector
}:
mkDerivation {
  pname = "inline-c";
  version = "0.7.0.1";
  src = ../inline-c/inline-c;
  isLibrary = true;
  isExecutable = false;
  libraryHaskellDepends = [
    ansi-wl-pprint base bytestring containers hashable mtl parsec
    parsers template-haskell transformers unordered-containers vector
  ];
  executableSystemDepends = [];
  testHaskellDepends = [
    ansi-wl-pprint base containers hashable hspec parsers QuickCheck
    raw-strings-qq regex-posix template-haskell transformers
    unordered-containers vector
  ];
  description = "Write Haskell source files including C code inline. No FFI required.";
  license = stdenv.lib.licenses.mit;
}
