# CMakeLists for muphfasa_powerspec (util/powerspec/). Only added when
# -Dmuphfasa_BUILD_POWERSPEC=ON, which also turns on AMReX's FFT module (FFTW).

add_executable(muphfasa_powerspec powerspec.cpp)

target_link_libraries(muphfasa_powerspec PRIVATE AMReX::amrex_${muphfasa_DIM}d)

if(AMReX_CUDA)
    setup_target_for_cuda_compilation(muphfasa_powerspec)
endif()
