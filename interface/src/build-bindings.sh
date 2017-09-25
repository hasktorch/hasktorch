#!/bin/bash -eu

# Build TorchStructs.hsc
c2hsc torch_structs.h

if [ "$(uname)" == "Darwin" ]; then
    ## OSX
    sed -i '' 's/<bindings.dsl.h>/"bindings.dsl.h"/g' TorchStructs.hsc
    sed -i '' 's/^#synonym_t.*//g' TorchStructs.hsc
    hsc2hs TorchStructs.hsc -o TorchStructs.hs
    sed -i '' '/.*LINE.*/d' TorchStructs.hs
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    ## Linux
    sed -i 's/<bindings.dsl.h>/"bindings.dsl.h"/g' TorchStructs.hsc
    sed -i 's/^#synonym_t.*//g' TorchStructs.hsc
    hsc2hs TorchStructs.hsc -o TorchStructs.hs
    sed -i '/.*LINE.*/d' TorchStructs.hs
fi

echo "Done"
