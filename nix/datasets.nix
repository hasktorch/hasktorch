{
  fetchurl,
  gzip,
  lib,
  linkFarm,
  symlinkJoin,
}: let
  attrsToList = attrs:
    map
    (name: {
      inherit name;
      value = attrs.${name};
    })
    (builtins.attrNames attrs);
in rec {
  mnist = linkFarm "mnist" (
    lib.pipe mnistParts [
      attrsToList
      (map (
        {
          name,
          value,
        }: {
          inherit name;
          path = value;
        }
      ))
    ]
  );
  mnistParts = let
    base = "https://ossci-datasets.s3.amazonaws.com/mnist";
  in {
    "train-images-idx3-ubyte.gz" = fetchurl {
      url = "${base}/train-images-idx3-ubyte.gz";
      hash = "sha256-RA/Kv3PMVG+iFHXoHqNwJlYF9WviEKQCTSyo8gNSNgk=";
      curlOptsList = ["-HUser-Agent: Wget/1.21.4"];
    };
    "train-labels-idx1-ubyte.gz" = fetchurl {
      url = "${base}/train-labels-idx1-ubyte.gz";
      hash = "sha256-NVJTSgpVi77WrtMrMMSVzKI9Vn7FLKyL4aBzDoAQJVw=";
      curlOptsList = ["-HUser-Agent: Wget/1.21.4"];
    };
    "t10k-images-idx3-ubyte.gz" = fetchurl {
      url = "${base}/t10k-images-idx3-ubyte.gz";
      hash = "sha256-jUIsewocHHkkWlvPB/6G4z7q/ueSuEWErsJ29aLbxOY=";
      curlOptsList = ["-HUser-Agent: Wget/1.21.4"];
    };
    "t10k-labels-idx1-ubyte.gz" = fetchurl {
      url = "${base}/t10k-labels-idx1-ubyte.gz";
      hash = "sha256-965g+S4A7G3r0jpgiMMdvSNx7KP/oN7677JZkkIErsY=";
      curlOptsList = ["-HUser-Agent: Wget/1.21.4"];
    };
  };
}
