#!/usr/bin/env bash
set -eu

# TorchStructs
function build_structs {
  local FILENAME="$1"
  local LIB="$2"
  case "$(uname -s)" in
    "Darwin"|"FreeBSD")
      ## Treat FreeBSD the same as OSX for now
      sed -i '' 's/<bindings.dsl.h>/"bindings.dsl.h"/g' ${FILENAME}.hsc
      sed -i '' 's/^#synonym_t.*//g' ${FILENAME}.hsc
      hsc2hs ${FILENAME}.hsc -o ${FILENAME}.hs
      sed -i '' '/.*LINE.*/d' ${FILENAME}.hs
      sed -i '' "s/${FILENAME}/Torch.Types.${LIB}.Structs/" ${FILENAME}.hsc
      mkdir -p Torch/Types/${LIB}
      mv ${FILENAME}.hs Torch/Types/${LIB}/Structs.hs
      rm ./${FILENAME}.hsc
      ;;

    "Linux")
      ## Linux
      sed -i 's/<bindings.dsl.h>/"bindings.dsl.h"/g' ${FILENAME}.hsc
      sed -i 's/^#synonym_t.*//g' ${FILENAME}.hsc
      hsc2hs ${FILENAME}.hsc -o ${FILENAME}.hs
      sed -i '/.*LINE.*/d' ${FILENAME}.hs
      sed -i "s/${FILENAME}/Torch.Types.${LIB}.Structs/" ${FILENAME}.hsc
      mkdir -p Torch/Types/${LIB}
      mv ${FILENAME}.hs Torch/Types/${LIB}/Structs.hs
      rm ./${FILENAME}.hsc
      ;;

    *)
      echo "Unknown OS"
      ;;
  esac
}

# Build TorchStructs.hsc
c2hsc th_structs.h
build_structs ThStructs TH

# Build Cuda structs
c2hsc thc_structs.h
build_structs ThcStructs THC

echo "Done"
