args@{ cudaVersion ? null, ...}:

let
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
  # hls = (import sources.hls-nix { }).hpkgs.haskell-language-server;
  shared = (
    import ./shared.nix {
      inherit (args);
      shellBuildInputs = [
        niv
        # hls
      ];
    }
  );
in
assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != null) || (cudaVersion == 10 && shared.defaultCuda102 != null);

if cudaVersion == null
then if shared.defaultCuda102 == null
  then shared.defaultCpu.shell
  else shared.defaultCuda102.shell
else if cudaVersion == 9
  then shared.defaultCuda92.shell
  else shared.defaultCuda102.shell
