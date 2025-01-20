
#
# Main component of TF Micro project 'person_detection'.
#
#include($ENV{IDF_PATH}/tools/cmake/project.cmake)
idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "insect_model_settings.cc"
        "1_model_data.cc"
        "2_model_data.cc"
        "3_model_data.cc"
        "4_model_data.cc"
        "5_model_data.cc"
        "6_model_data.cc"
        "7_model_data.cc"
        "8_model_data.cc"
        "person_detect_model_data.cc"
        "insect_detect_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"

    PRIV_REQUIRES console static_images spi_flash
    INCLUDE_DIRS "")
 
if(NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is not defined. Use 'idf.py build -DMODEL=<number>' to set the model.")
else()
    message(STATUS "Building with MODEL=${MODEL}")
    add_compile_definitions(MODEL=${MODEL})
endif()