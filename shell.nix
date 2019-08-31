{ pkgs ?
   let
     # revision: https://github.com/NixOS/nixpkgs/pull/65041
     rev = "9420c33455c5b3e0876813bdd739bc75b3ccbd8a";
     url = "https://github.com/NixOS/nixpkgs/archive/${rev}.tar.gz";
     # nix-prefetch-url --unpack ${url}
     sha256 = "0l82qfcmip0mfijcxssaliyxayfh0c11bk66yhwna1ixc0s0jjd2";
     nixpkgs = builtins.fetchTarball { inherit url sha256; };
   in
     import nixpkgs { config.allowUnfree = true; config.allowUnsupportedSystem = true; }
}:

with pkgs;

let

  hsenv = haskellPackages.ghcWithPackages (p: with p; [
    cabal-install
    ansi-wl-pprint
    async
    bytestring
    containers
    exceptions
    hashable
    mtl
    optparse-applicative
    parsec
    parsers
    safe-exceptions
    sysinfo
    template-haskell
    transformers
    transformers-compat
    unordered-containers
    vector
  ]);

in

stdenv.mkDerivation {
  name = "hasktorch-dev";
  buildInputs = [ hsenv mkl python3Packages.pytorchWithoutCuda ];
  shellHook = ''
    export CPATH=${python3Packages.pytorchWithoutCuda}/lib/${python3Packages.python.libPrefix}/site-packages/torch/include/torch/csrc/api/include
  '';

}
