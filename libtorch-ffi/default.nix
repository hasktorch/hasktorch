{ mkDerivation, async, base, bytestring, containers, hspec
, hspec-discover, inline-c, inline-c-cpp, mkl
, optparse-applicative, safe-exceptions, stdenv, sysinfo
, template-haskell, torch
}:
mkDerivation {
  pname = "libtorch-ffi";
  version = "1.1.0.0";
  src = ./.;
  libraryHaskellDepends = [
    async base bytestring containers inline-c inline-c-cpp
    optparse-applicative safe-exceptions sysinfo template-haskell
  ];
  librarySystemDepends = [ mkl torch ];
  testHaskellDepends = [
    base containers hspec hspec-discover inline-c inline-c-cpp
    optparse-applicative safe-exceptions
  ];

  configureFlags = [ "--extra-include-dirs=${torch}/include/torch/csrc/api/include" ];

  doHaddock = false;

  homepage = "https://github.com/hasktorch/hasktorch#readme";
  description = "test out alternative options for ffi interface to libtorch 1.x";
  license = stdenv.lib.licenses.bsd3;
}
