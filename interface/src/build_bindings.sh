#!/bin/bash -eu

c2hsc torch_structs.h

mv TorchStructs.hsc tmp.hsc

sed 's/<bindings.dsl.h>/"bindings.dsl.h"/g' tmp.hsc > tmp2.hsc

sed 's/^#synonym_t.*//g' tmp2.hsc > TorchStructs.hsc

hsc2hs TorchStructs.hsc -o TorchStructs.hs

# sed 's/^{-# LINE .*\n//g' tmp3.hs > TorchStructs.hs

rm tmp.hsc

rm tmp2.hsc
