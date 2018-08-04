{ mkDerivation, base, bytestring, cryptonite, dimensions, directory
, filepath, hasktorch-core, HTTP, microlens, network-uri
, singletons, stdenv
}:
mkDerivation {
  pname = "hasktorch-examples";
  version = "0.0.1.0";
  src = ./.;
  isLibrary = false;
  isExecutable = true;
  executableHaskellDepends = [
    base bytestring cryptonite dimensions directory filepath
    hasktorch-core HTTP microlens network-uri singletons
  ];
  homepage = "https://github.com/austinvhuang/hasktorch#readme";
  description = "Torch for tensors and neural networks in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
