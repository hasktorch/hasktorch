let sources = import ./nix/sources.nix; in
{ haskellNix ? import sources.haskell-nix { sourcesOverride = sources; }
, haskellNixOverlays ? (pkgsNew: pkgsOld: { haskell-nix = pkgsOld.haskell-nix // {
    hackageSrc = sources.hackage-nix;
    stackageSrc = sources.stackage-nix;
  }; })
, nixpkgsSrc ? haskellNix.sources.nixpkgs-2003
, haskellCompiler ? "ghc8101"
, shellBuildInputs ? []
, ...
}:
let
  libtorchSrc = pkgs:
    pkgs.callPackage "${sources.pytorch-world}/libtorch/release.nix" { };

  libtorchOverlayCpu = pkgsNew: pkgsOld:
    let libtorch = (libtorchSrc pkgsOld).libtorch_cpu; in
    {
      c10 = libtorch;
      torch = libtorch;
      torch_cpu = libtorch;
    };

  libtorchOverlayCuda92 = pkgsNew: pkgsOld:
    let libtorch = (libtorchSrc pkgsOld).libtorch_cudatoolkit_9_2; in
    {
      c10 = libtorch;
      torch = libtorch;
      torch_cpu = libtorch;
      torch_cuda = libtorch;
    };

  libtorchOverlayCuda102 = pkgsNew: pkgsOld:
    let libtorch = (libtorchSrc pkgsOld).libtorch_cudatoolkit_10_2; in
    {
      c10 = libtorch;
      torch = libtorch;
      torch_cpu = libtorch;
      torch_cuda = libtorch;
    };

  default =
    { overlays
    , pkgs ? (import nixpkgsSrc (haskellNix.nixpkgsArgs // { overlays = haskellNix.overlays ++ [ haskellNixOverlays ] ++ overlays; }))
    , USE_CUDA ? false
    , USE_GCC ? !USE_CUDA && pkgs.stdenv.hostPlatform.system == "x86_64-darwin"
    }: if USE_CUDA && pkgs.stdenv.hostPlatform.system == "x86_64-darwin" then { } else rec {
      inherit pkgs;
      hsPkgs = pkgs.haskell-nix.cabalProject {
        src = pkgs.haskell-nix.haskellLib.cleanGit { name = "hasktorch"; src = ./.; };
        # ghc = pkgs.buildPackages.pkgs.haskell-nix.compiler.${haskellCompiler};
        compiler-nix-name = haskellCompiler;
        modules = [
          { reinstallableLibGhc = true; }
          ({ config, ... }: {
            reinstallableLibGhc = true;
            packages.codegen.components.tests.doctests.doCheck = false;
            packages.codegen.doHaddock = false;
            packages.libtorch-ffi.configureFlags = [
              "--extra-lib-dirs=${pkgs.torch}/lib"
              "--extra-include-dirs=${pkgs.torch}/include"
              "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
            ];
            packages.libtorch-ffi.flags.cuda = USE_CUDA;
            packages.libtorch-ffi.flags.gcc = USE_GCC;
            packages.examples.doHaddock = false;
            packages.experimental.doHaddock = false;
          })
        ];
      };
      shell = hsPkgs.shellFor {
        withHoogle = true;
        tools = {
          # cabal = "3.2.0.0";
          ghcide = "0.2.0";
        };
        nativeBuildInputs = [(pkgs.haskell-nix.cabalProject {
          src = sources.cabal;
          modules = [ { reinstallableLibGhc = true; } ];
        }).cabal-install.components.exes.cabal];  
        buildInputs = shellBuildInputs;
        exactDeps = true; # false; # set to true as soon as haskell.nix issue #231 is resolved
        shellHook = ''
          export CPATH=${pkgs.torch}/include/torch/csrc/api/include
        '';
      };
    };

  defaultCpu = default { overlays = [ libtorchOverlayCpu ]; };
  defaultCuda92 = default { overlays = [ libtorchOverlayCuda92 ]; USE_CUDA = true; };
  defaultCuda102 = default { overlays = [ libtorchOverlayCuda102 ]; USE_CUDA = true; };

in

{

  inherit defaultCpu;
  inherit defaultCuda92;
  inherit defaultCuda102;

}
