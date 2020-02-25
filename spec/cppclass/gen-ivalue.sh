#!/bin/bash

cat <<EOF > spec/cppclass/ivalue.yaml
signature: IValue
cppname: at::IValue
hsname: IValue
functions: []
constructors:
- new() -> IValue
- new(IValue x) -> IValue
methods:
EOF

cat deps/libtorch/include/ATen/core/ivalue.h \
  | grep -vi c10 \
  | grep -vi TensorImpl \
  | perl -ne 'BEGIN{$f=0;}; {if (/struct CAFFE2_API IValue/){$f=1;}; if (/^};/ || /struct CAFFE2_API WeakIValue/){print $_;$f=0;}; if($f==1){print $_;}};' \
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
  >> spec/cppclass/ivalue.yaml
