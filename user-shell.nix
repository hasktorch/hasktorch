## User shell for using hasktorch as a user

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
  pytorch = python3Packages.pytorchWithoutCuda.override {
    mklSupport = true; inherit mkl;
  };

  myHaskellPackages = haskellPackages.override {
    overrides = self: super: {
      inline-c = self.callPackage ./nix/inline-c.nix {};
      inline-c-cpp = self.callPackage ./nix/inline-c-cpp.nix {};
      libtorch-ffi = self.callPackage ./libtorch-ffi {
        torch = pytorch.dev;
      };
      hasktorch = self.callPackage ./hasktorch {};
    };
  };

  hsenv = myHaskellPackages.ghcWithPackages (p: with p; [
    inline-c
    inline-c-cpp
    libtorch-ffi
    hasktorch
    # add any dependencies for your project
    random
    call-stack
    hspec
  ]);

in

stdenv.mkDerivation {
  name = "hasktorch-env";
  buildInputs = [ hsenv pytorch.dev mkl ];

  shellHook = ''
  '';

}
