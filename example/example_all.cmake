
include_directories(${CMAKE_HOME_DIRECTORY}/include)

AUX_SOURCE_DIRECTORY(example/clean_text GPT_SOVITS_CPP_TEST_CLEAN_TEXT_SOURCE)
add_executable(gpt_sovits_cpp_test_clean_text ${GPT_SOVITS_CPP_TEST_CLEAN_TEXT_SOURCE})
target_link_libraries(gpt_sovits_cpp_test_clean_text PUBLIC gsv_lib)


if (USE_ONNX)
  add_executable(gpt_sovits_cpp_test_bert example/model/bert.cpp)
  target_link_libraries(gpt_sovits_cpp_test_bert PUBLIC gsv_lib)
endif ()


if (USE_ONNX)
  add_executable(gpt_sovits_cpp_cloud_create_onnx example/onnx/cloud_create_speaker.cpp)
  target_link_libraries(gpt_sovits_cpp_cloud_create_onnx PUBLIC gsv_lib)

  add_executable(gpt_sovits_cpp_edge_inference_onnx example/onnx/edge_inference.cpp)
  target_link_libraries(gpt_sovits_cpp_edge_inference_onnx PUBLIC gsv_lib)

  add_executable(gpt_sovits_cpp_multi_speaker_onnx example/onnx/multi_speaker.cpp)
  target_link_libraries(gpt_sovits_cpp_multi_speaker_onnx PUBLIC gsv_lib)

  add_executable(gpt_sovits_cpp_streaming_onnx example/onnx/streaming_inference.cpp)
  target_link_libraries(gpt_sovits_cpp_streaming_onnx PUBLIC gsv_lib)
endif ()

if (USE_TENSORRT)
  add_executable(gpt_sovits_cpp_cloud_create_trt example/tensorrt/cloud_create_speaker.cpp)
  target_link_libraries(gpt_sovits_cpp_cloud_create_trt PUBLIC gsv_lib)

  add_executable(gpt_sovits_cpp_edge_inference_trt example/tensorrt/edge_inference.cpp)
  target_link_libraries(gpt_sovits_cpp_edge_inference_trt PUBLIC gsv_lib)

  add_executable(gpt_sovits_cpp_streaming_trt example/tensorrt/streaming_inference.cpp)
  target_link_libraries(gpt_sovits_cpp_streaming_trt PUBLIC gsv_lib)

  # NVTX profiling support
  if (USE_NVTX)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(gpt_sovits_cpp_edge_inference_trt PRIVATE USE_NVTX)
    target_link_libraries(gpt_sovits_cpp_edge_inference_trt PRIVATE CUDA::nvtx3)
    target_compile_definitions(gpt_sovits_cpp_streaming_trt PRIVATE USE_NVTX)
    target_link_libraries(gpt_sovits_cpp_streaming_trt PRIVATE CUDA::nvtx3)
  endif()

endif ()


target_copy_res(gpt_sovits_cpp_test_clean_text)
if (USE_ONNX)
  target_copy_res(gpt_sovits_cpp_test_bert)
endif ()

if (USE_ONNX)
  target_copy_res(gpt_sovits_cpp_cloud_create_onnx)
  target_copy_res(gpt_sovits_cpp_edge_inference_onnx)
  target_copy_res(gpt_sovits_cpp_multi_speaker_onnx)
  target_copy_res(gpt_sovits_cpp_streaming_onnx)
endif ()

if (WIN32 AND COMMAND auto_copy_backend_dlls)
  if (USE_ONNX)
    auto_copy_backend_dlls(gpt_sovits_cpp_test_clean_text)
    auto_copy_backend_dlls(gpt_sovits_cpp_test_bert)
    auto_copy_backend_dlls(gpt_sovits_cpp_test_pipline_cpp)

    auto_copy_backend_dlls(gpt_sovits_cpp_cloud_create_onnx)
    auto_copy_backend_dlls(gpt_sovits_cpp_edge_inference_onnx)
    auto_copy_backend_dlls(gpt_sovits_cpp_multi_speaker_onnx)
    auto_copy_backend_dlls(gpt_sovits_cpp_streaming_onnx)
  endif ()
  if (USE_ONNX)
    auto_copy_backend_dlls(gpt_sovits_cpp_cloud_create_trt)
    auto_copy_backend_dlls(gpt_sovits_cpp_edge_inference_trt)
    auto_copy_backend_dlls(gpt_sovits_cpp_streaming_trt)
  endif ()
endif ()

