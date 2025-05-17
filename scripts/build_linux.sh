rm -rf build
mkdir build && cd build
# 编译参数：
builder="-G Ninja"

if [ "$1" == "make" ]; then
builder=""
fi

SM="90a"

cmake ${builder} .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${SM}

cmake --build . -- -j$(nproc)
# cmake --install .