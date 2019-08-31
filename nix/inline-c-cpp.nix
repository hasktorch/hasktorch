{ mkDerivation, base, containers, hspec, inline-c, safe-exceptions
, stdenv, template-haskell
}:
mkDerivation {
  pname = "inline-c-cpp";
  version = "0.3.0.1";
  src = ../inline-c/inline-c-cpp;
  libraryHaskellDepends = [
    base containers inline-c safe-exceptions template-haskell
  ];
  testHaskellDepends = [
    base containers hspec inline-c safe-exceptions
  ];
  description = "Lets you embed C++ code into Haskell";
  license = stdenv.lib.licenses.mit;
}
