cmake_minimum_required(VERSION 3.2)


set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

project( ocr_classifier )

find_package( OpenCV REQUIRED )
find_package( Tesseract REQUIRED )


include_directories(extern/)
include_directories(${Tesseract_INCLUDE_DIRS})
include_directories(${Leptonica_INCLUDE_DIRS})

add_executable( ocr_classifier main.cpp)

target_compile_features( ocr_classifier PRIVATE cxx_std_17)

target_link_libraries( ocr_classifier ${OpenCV_LIBS} libtesseract ${Leptonica_LIBRARIES})