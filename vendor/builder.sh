source "$stdenv"/setup
BUILD=build2

if nvcc --version &> /dev/null; then
  sans_cuda=false
else
  sans_cuda=true
fi

# Building in a read-only filesystem? I think not!
cp --recursive "$src" ./
chmod --recursive u=rwx ./"$(basename "$src")"
cd ./"$(basename "$src")"

# continue!
mkdir -p ./$BUILD && cd ./$BUILD

echo "exclude cuda? $sans_cuda"
cmake .. -DNO_CUDA=$sans_cuda -Wno-dev -DCMAKE_INSTALL_PREFIX=./$BUILD
make install

echo "ls $out" && ls $out
echo "ls $src" && ls $src
echo "ls ./$(basename "$src")" && ls ./"$(basename "$src")"
echo "ls ./$BUILD" && ls ./$BUILD
