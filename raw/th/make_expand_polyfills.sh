#!/usr/bin/env bash

POLYFILL=cbits/expand_polyfill.c
echo '#include "TH/THTensor.h"'  >  $POLYFILL
echo '#include "TH/THStorage.h"' >> $POLYFILL
echo ''                          >> $POLYFILL

for type in Byte Char Short Int Long Half Float Double; do
  cat cbits/expand_template.in | sed "s/<<TYPE>>/$type/g"      >> $POLYFILL
  echo "\n\n\\\\ Polyfilling expand functions for $type: \n\n" >> $POLYFILL
done


