let sources = import ./sources.nix; in
{ haskellNix ? import sources.haskell-nix { sourcesOverride = sources; }
, haskellNixOverlays ? (pkgsNew: pkgsOld: { haskell-nix = pkgsOld.haskell-nix // { hackageSrc = sources.hackage-nix; }; })
, nixpkgsSrc ? haskellNix.sources.nixpkgs-2003
, haskellCompiler ? "ghc883"
, shellBuildInputs ? []
, ...
}:
let
  libtorchSrc = pkgs:
    let src = pkgs.fetchFromGitHub {
      owner = "stites";
      repo = "pytorch-world";
      rev = "6dc929a791918fff065bb40cbc3db8a62beb2a30";
      sha256 = "140a2l1l1qnf7v2s1lblrr02mc0knsqpi06f25xj3qchpawbjd4c";
    };
    in (pkgs.callPackage "${src}/libtorch/release.nix" { });

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
        compiler-nix-name = haskellCompiler;
        modules = [
          ({ config, ... }: {
            packages.libtorch-ffi.configureFlags = [
              "--extra-lib-dirs=${pkgs.torch}/lib"
              "--extra-include-dirs=${pkgs.torch}/include"
              "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
            ];
            packages.libtorch-ffi.flags.cuda = USE_CUDA;
            packages.libtorch-ffi.flags.gcc = USE_GCC;
          })
        ];
      };
      shell = hsPkgs.shellFor {
        withHoogle = true;
        tools = { cabal = "3.2.0.0"; ghcide = "0.2.0"; };
        buildInputs = shellBuildInputs;
        exactDeps = false; # set to true as soon as haskell.nix issue #231 is resolved
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
