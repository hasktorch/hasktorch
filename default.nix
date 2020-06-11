let sources = import ./nix/sources.nix; in
{ haskellNix ? import sources.haskell-nix { sourcesOverride = sources; }
, haskellNixOverlays ? (pkgsNew: pkgsOld: { haskell-nix = pkgsOld.haskell-nix // { hackageSrc = sources.hackage-nix; }; })
, nixpkgsSrc ? haskellNix.sources.nixpkgs-2003
, haskellCompiler ? "ghc883"
, shellBuildInputs ? []
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
        tools = { cabal = "3.2.0.0"; };
        buildInputs = shellBuildInputs;
        exactDeps = true;
        shellHook = ''
          export CPATH=${pkgs.torch}/include/torch/csrc/api/include
        '';
      };
    };

  defaultCpu = default { overlays = [ libtorchOverlayCpu ]; };
  defaultCuda92 = default { overlays = [ libtorchOverlayCuda92 ]; USE_CUDA = true; };
  defaultCuda102 = default { overlays = [ libtorchOverlayCuda102 ]; USE_CUDA = true; };

  # f = x: y: if x == null then null else y;
  # g = x: y: if x == null then null else x.y;
in
{
  inherit defaultCpu;
  inherit defaultCuda92;
  inherit defaultCuda102;

  # libtorchFFICpu = hsPkgsCpu.libtorch-ffi;
  # ${f hsPkgsCuda92 "libtorchFFICuda92"} = g hsPkgsCuda92 "libtorch-ffi";
  # ${f hsPkgsCuda102 "libtorchFFICuda102"} = g hsPkgsCuda102 "libtorch-ffi";
  # libtorchFFICuda102 = hsPkgsCuda102.libtorch-ffi;
  # hasktorchCpu = hsPkgsCpu.hasktorch;
}


# let
#   shared = import ./nix/shared.nix { };
# in
#   { inherit (shared)
#       hasktorch-codegen
#       libtorch-ffi_cpu
#       hasktorch_cpu
#       hasktorch-examples_cpu
#       hasktorch-experimental_cpu
#       hasktorch-docs
#     ;

#     ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_9_2"} = shared.libtorch-ffi_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch_cudatoolkit_9_2"} = shared.hasktorch_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_9_2"} = shared.hasktorch-examples_cudatoolkit_9_2;
#     ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_9_2"} = shared.hasktorch-experimental_cudatoolkit_9_2;

#     ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_10_2"} = shared.libtorch-ffi_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch_cudatoolkit_10_2"} = shared.hasktorch_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_10_2"} = shared.hasktorch-examples_cudatoolkit_10_2;
#     ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_10_2"} = shared.hasktorch-experimental_cudatoolkit_10_2;
#   }
