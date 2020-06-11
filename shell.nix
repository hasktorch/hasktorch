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
assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != { }) || (cudaVersion == 10 && shared.defaultCuda102 != { });

if cudaVersion == null
then if shared.defaultCuda102 == { }
  then shared.defaultCpu.shell
  else shared.defaultCuda102.shell
else if cudaVersion == 9
  then shared.defaultCuda92.shell
  else shared.defaultCuda102.shell
