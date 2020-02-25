#!/bin/bash

cat <<EOF > spec/cppclass/tensor.yaml
signature: Tensor
cppname: at::Tensor
hsname: Tensor
functions: []
constructors:
- new() -> Tensor
- new(Tensor x) -> Tensor
methods:
EOF

cat  deps/libtorch/include/ATen/core/TensorBody.h \
  | grep -vi c10 \
  | grep -vi TensorImpl \
  | perl -ne 'BEGIN{$f=0;}; {if (/class Tensor/){$f=1;}; if (/^};/ || /Tensor alias/){print $_;$f=0;}; if($f==1){print $_;}};' \
  | grep '^  [^ /].*(.*)' \
  | sed -e 's/).*/)/g' \
  | sed -e 's/const //g' \
  | sed -e 's/inline //g' \
  | sed -e 's/&//g' \
  | sed -e 's/^ *//g' \
  | sed -e 's/^char \*/char*/g' \
  | sed -e 's/^T \*/T*/g' \
  | sed -e 's/ *\([^ ]*\) \(.*\)$/\2 -> \1/g' \
  | sed -e 's/^ *//g' \
  | sed -e 's/^/- /g' \
  >> spec/cppclass/tensor.yaml
