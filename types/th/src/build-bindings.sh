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
      sed -i '' "s/${FILENAME}/Structs/" ${FILENAME}.hs
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
      sed -i -e "s/${FILENAME}/Structs/" ${FILENAME}.hs
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
c2hsc --gcc $(command -v gcc) --prefix Torch.Types.TH --verbose th_structs.h
build_structs ThStructs TH

if command -v nvcc &> /dev/null; then
  # Build Cuda structs
  c2hsc --gcc $(command -v nvcc) --prefix Torch.Types.THC --verbose thc_structs.h
  build_structs ThcStructs THC
fi

echo "Done"
