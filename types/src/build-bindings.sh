#!/usr/bin/env bash
set -eu

# Build TorchStructs.hsc
c2hsc torch_structs.h

case "$(uname -s)" in
  "Darwin"|"FreeBSD")
    ## Treat FreeBSD the same as OSX for now
    sed -i '' 's/<bindings.dsl.h>/"bindings.dsl.h"/g' TorchStructs.hsc
    sed -i '' 's/^#synonym_t.*//g' TorchStructs.hsc
    hsc2hs TorchStructs.hsc -o TorchStructs.hs
    sed -i '' '/.*LINE.*/d' TorchStructs.hs
    sed -i '' 's/TorchStructs/Torch.Types.Structs/' TorchStructs.hsc
    mkdir -p Torch/Types
    mv TorchStructs.hs Torch/Types/Structs.hs
    rm ./TorchStructs.hsc
    ;;

  "Linux")
    ## Linux
    sed -i 's/<bindings.dsl.h>/"bindings.dsl.h"/g' TorchStructs.hsc
    sed -i 's/^#synonym_t.*//g' TorchStructs.hsc
    hsc2hs TorchStructs.hsc -o TorchStructs.hs
    sed -i '/.*LINE.*/d' TorchStructs.hs
    sed -i 's/TorchStructs/Torch.Types.Structs/' TorchStructs.hsc
    mkdir -p Torch/Types
    mv TorchStructs.hs Torch/Types/Structs.hs
    rm ./TorchStructs.hsc
    ;;

  *)
    echo "Unknown OS"
    exit 1
    ;;
esac

echo "Done"
