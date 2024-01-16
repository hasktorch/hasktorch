{
  fetchurl,
  gzip,
  lib,
  linkFarm,
  symlinkJoin,
}:

let
  attrsToList =
    attrs:
    map
      (name: {
        inherit name;
        value = attrs.${name};
      })
      (builtins.attrNames attrs);
  fetchgz =
    {
      url ? builtins.head urls,
      urls ? [ ],
      postFetch ? "",
      ...
    }@args:
    fetchurl (
      args
      // {
        name = lib.removeSuffix ".gz" (baseNameOf url);
        downloadToTemp = true;
        postFetch =
          ''
            zcat "$downloadedFile" > $out
          ''
          + postFetch;
      }
    );
in
rec {
  mnist = linkFarm "mnist" (
    lib.pipe mnistParts [
      attrsToList
      (map (
        { name, value }:
        {
          inherit name;
          path = value;
        }
      ))
    ]
  );
  mnistParts = {
    train-images-idx3-ubyte = fetchgz {
      url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
      hash = "sha256-uokQRuZQXXqty74laAoHOK0Wrsk73n+bZeh6L8JXdts=";
    };
    train-labels-idx1-ubyte = fetchgz {
      url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
      hash = "sha256-ZaUMu/TpBtcIMoeK2FzNpTM6l/D0w90u8JqKnu9xAcU=";
    };
    t10k-images-idx3-ubyte = fetchgz {
      url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
      hash = "sha256-D6eJjVCSeeSClY6M6ByOd9s/L4JU4mZhzrd2LE1JTOc=";
    };
    t10k-labels-idx1-ubyte = fetchgz {
      url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
      hash = "sha256-/3vP1BbeM3MaMIw/JmzDUSIsNImOy+r4R/BuSPfsM/I=";
    };
  };
}
