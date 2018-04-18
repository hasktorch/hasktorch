#!/usr/bin/env bash

POLYFILL=cbits/expand_polyfill.c
echo '#include "TH/THTensor.h"'  >  $POLYFILL
echo '#include "TH/THStorage.h"' >> $POLYFILL
echo ''                          >> $POLYFILL

for type in Byte Char Short Int Long Half Float Double; do
  cat cbits/expand_template.in | sed "s/<<TYPE>>/$type/g"  >> $POLYFILL
  echo ""                                                  >> $POLYFILL
  echo ""                                                  >> $POLYFILL
  echo "// Polyfilling expand functions for $type:"        >> $POLYFILL
  echo ""                                                  >> $POLYFILL
  echo ""                                                  >> $POLYFILL

  mkdir -p ./polyfill/Torch/FFI/TH/${type}/
  FILE="./polyfill/Torch/FFI/TH/${type}/Expand.hs"
  rm -rf $FILE
  touch $FILE

  echo '{-# LANGUAGE ForeignFunctionInterface #-}'                                                                   >  $FILE
  echo "module Torch.FFI.TH.${type}.Expand where"                                                                    >> $FILE
  echo ''                                                                                                            >> $FILE
  echo 'import Foreign'                                                                                              >> $FILE
  echo 'import Data.Word'                                                                                            >> $FILE
  echo 'import Torch.Types.TH'                                                                                       >> $FILE
  echo ''                                                                                                            >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "foreign import ccall \"&TH${type}Tensor_expand\""                                                            >> $FILE
  echo "  c_expand_ :: Ptr C'TH${type}Tensor -> Ptr C'TH${type}Tensor -> Ptr C'THLongStorage -> IO ()"               >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "c_expand :: Ptr C'THState -> Ptr C'TH${type}Tensor -> Ptr C'TH${type}Tensor -> Ptr C'THLongStorage -> IO ()" >> $FILE
  echo "c_expand = const c_expand_"                                                                                  >> $FILE
  echo ''                                                                                                            >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "foreign import ccall \"&TH${type}Tensor_expandNd\""                                                          >> $FILE
  echo "  c_expandNd_ :: Ptr (Ptr C'TH${type}Tensor) -> Ptr (Ptr C'TH${type}Tensor) -> CInt -> IO ()"                >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "c_expandNd :: Ptr C'THState -> Ptr (Ptr C'TH${type}Tensor) -> Ptr (Ptr C'TH${type}Tensor) -> CInt -> IO ()"  >> $FILE
  echo "c_expandNd = const c_expandNd_"                                                                              >> $FILE
  echo ''                                                                                                            >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "foreign import ccall \"&TH${type}Tensor_newExpand\""                                                         >> $FILE
  echo "  c_newExpand_ :: Ptr C'TH${type}Tensor -> Ptr C'THLongStorage -> IO (Ptr C'TH${type}Tensor)"                >> $FILE
  echo ''                                                                                                            >> $FILE
  echo "c_newExpand :: Ptr C'THState -> Ptr C'TH${type}Tensor -> Ptr C'THLongStorage -> IO (Ptr C'TH${type}Tensor)"  >> $FILE
  echo "c_newExpand = const c_newExpand_"                                                                            >> $FILE
  echo ''                                                                                                            >> $FILE
done


