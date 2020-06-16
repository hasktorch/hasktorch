args@{ cudaVersion ? null, ... }:

let
  shared = (import ./shared.nix { inherit (args); });
  default = (
    if cudaVersion == null
    then shared.defaultCpu
    else if cudaVersion == 9
      then shared.defaultCuda92
      else shared.defaultCuda102
  );
  pkgs = default.pkgs;
  hsPkgs = default.hsPkgs;
  getComponents = group: name:
    pkgs.haskell-nix.haskellLib.collectComponents group (package: (builtins.tryEval (package.identifier.name == name)).value) hsPkgs;
in
  assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != { }) || (cudaVersion == 10 && shared.defaultCuda102 != { });
  {
    codegen = (getComponents "exes" "codegen").codegen;
    libtorch-ffi = (getComponents "library" "libtorch-ffi").libtorch-ffi;
    hasktorch = (getComponents "library" "hasktorch").hasktorch;
    examples = (getComponents "exes" "examples").examples;
    experimental = (getComponents "exes" "experimental").experimental;

    combined-haddock = let
      haddock-combine = pkgs.callPackage ./nix/haddock-combine.nix {
        runCommand = pkgs.runCommand;
        lib = pkgs.lib;
        ghc = hsPkgs.ghcWithPackages (ps: []);
      };
      projectPackages = pkgs.haskell-nix.haskellLib.selectProjectPackages hsPkgs;
      toHaddock = pkgs.haskell-nix.haskellLib.collectComponents' "library" projectPackages;
      in haddock-combine {
        hspkgs = builtins.attrValues toHaddock;
        prologue = pkgs.writeTextFile {
          name = "prologue";
          text = "Documentation for hasktorch and its libraries.";
        };
      };
  }
