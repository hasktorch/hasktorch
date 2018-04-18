#!/usr/bin/env bash
# set -eu

# build structs
function build_structs {
  local FILENAME="$1"
  local LIB="$2"
  case "$(uname -s)" in

    # Treat FreeBSD the same as OSX for now
    "Darwin"|"FreeBSD")
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
      sed -i 's/<bindings.dsl.h>/"bindings.dsl.h"/g' ${FILENAME}.hsc
      sed -i 's/^#synonym_t.*//g' ${FILENAME}.hsc
      hsc2hs ${FILENAME}.hsc -o ${FILENAME}.hs
      sed -i '/.*LINE.*/d' ${FILENAME}.hs
      sed -i -e "s/${FILENAME}/${LIB}.Structs/" ${FILENAME}.hs
      mkdir -p Torch/Types/${LIB}
      mv ${FILENAME}.hs Torch/Types/${LIB}/Structs.hs
      rm ./${FILENAME}.hsc
      ;;

    *)
      echo "Unknown OS"
      ;;

  esac
}

function build_thc {
  # Build Cuda structs
  if command -v nvcc &> /dev/null; then
    cd thc/src
    c2hsc --gcc $(command -v gcc) --prefix Torch.Types --verbose thc_cuda_runtime_types.h
    build_structs ThcCudaRuntimeTypes Cuda

    c2hsc --gcc $(command -v gcc) --prefix Torch.Types --verbose thc_curand_types.h
    build_structs ThcCurandTypes CuRand

    c2hsc --gcc $(command -v gcc) --prefix Torch.Types --verbose thc_structs.h
    build_structs ThcStructs THC

  else
    echo "can't find nvcc -- please make sure cuda is installed"
    exit 1
  fi
}

function build_th {
  cd th/src
  c2hsc --gcc $(command -v gcc) --prefix Torch.Types --verbose th_structs.h
  build_structs ThStructs TH
}

if [[ "${0:-}" != "${BASH_SOURCE:-}" ]]; then
  printf 'this script is not meant to be sourced.'
  printf ' Please run `./built-structs.sh` instead or'
  printf '`chmod u+x ./built-structs.sh && ./built-structs.sh`\n'
else
  # Build TorchStructs.hsc
  case "${1:-}" in
    "all")
      build_th
      build_thc
      ;;
    "thc") build_thc ;;
    "th")  build_th ;;
    "--help"|"-h"|"help"|*)
      echo "./build-bindings.sh -- build torch structs from the hasktorch/types subdirectory."
      echo ""
      echo "USAGE:"
      echo "  ./build-bindings.sh {all|thc|th}      build structs for types/th, types/thc, or both projects"
      echo "  ./build-bindings.sh {--help|-h|help}  short this prompt"
      echo ""
      ;;
  esac
fi
