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
  default = (
    if cudaVersion == null
    then shared.defaultCpu
    else if cudaVersion == 9
      then shared.defaultCuda92
      else shared.defaultCuda102
  );
in
  assert cudaVersion == null || (cudaVersion == 9 && shared.defaultCuda92 != { }) || (cudaVersion == 10 && shared.defaultCuda102 != { });
  default.shell
