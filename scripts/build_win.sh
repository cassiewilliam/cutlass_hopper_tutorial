rm -rf build
mkdir build && cd build
 
SM="89"
 
export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
cmake .. -A x64 -T "v143,cuda=$CUDA_PATH_V12_5" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${SM}
 
cmake --build .
cmake --install .