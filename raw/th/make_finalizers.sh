#!/usr/bin/env bash

for type in Int Byte Char Short Long Half Float Double; do
  for ctype in Tensor Storage; do
    mkdir -p ./polyfill/Torch/FFI/TH/${type}/
    FILE="./polyfill/Torch/FFI/TH/${type}/Free${ctype}.hs"
    rm -rf $FILE
    touch $FILE

    echo '{-# LANGUAGE ForeignFunctionInterface #-}'                              >  $FILE
    echo "module Torch.FFI.TH.${type}.Free${ctype} where"                         >> $FILE
    echo ''                                                                       >> $FILE
    echo 'import Foreign'                                                         >> $FILE
    echo 'import Data.Word'                                                       >> $FILE
    echo 'import Torch.Types.TH'                                                  >> $FILE
    echo ''                                                                       >> $FILE
    echo "foreign import ccall \"&free_${type}${ctype}\""                         >> $FILE
    echo "  p_free :: FunPtr (Ptr C'THState -> Ptr C'TH${type}${ctype} -> IO ())" >> $FILE
    echo ''                                                                       >> $FILE
  done
done


