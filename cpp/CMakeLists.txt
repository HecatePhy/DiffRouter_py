cmake_minimum_required(VERSION 3.17)
project(diffrouter_rrg_extract CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)

# Cap'n Proto: prefer POTTER_ROOT bundled lib, else system, else FetchContent
set(POTTER_ROOT "$ENV{POTTER_ROOT}" CACHE PATH "Path to Potter repo with libs/capnproto")
if(NOT POTTER_ROOT AND EXISTS "/home/usr1/zhilix/Potter")
    set(POTTER_ROOT "/home/usr1/zhilix/Potter")
endif()

if(POTTER_ROOT AND EXISTS "${POTTER_ROOT}/libs/capnproto/CMakeLists.txt")
    message(STATUS "Using Cap'n Proto from POTTER_ROOT=${POTTER_ROOT}")
    add_subdirectory("${POTTER_ROOT}/libs/capnproto" "${CMAKE_BINARY_DIR}/capnproto" EXCLUDE_FROM_ALL)
    set(CAPNP_INCLUDE "${POTTER_ROOT}/libs/capnproto/c++/src")
else()
    find_package(CapnProto CONFIG QUIET)
    if(NOT CapnProto_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            capnproto
            GIT_REPOSITORY https://github.com/capnproto/capnproto.git
            GIT_TAG v1.0.2
        )
        set(CAPNP_TESTING OFF CACHE BOOL "" FORCE)
        set(CAPNP_TOOLS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(capnproto)
    endif()
    set(CAPNP_INCLUDE "${capnproto_SOURCE_DIR}/c++/src")
endif()

set(SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schema/interchange")
set(CAPNP_SCHEMA_SOURCES
    "${SCHEMA_DIR}/References.capnp.c++"
    "${SCHEMA_DIR}/LogicalNetlist.capnp.c++"
    "${SCHEMA_DIR}/DeviceResources.capnp.c++"
)

add_executable(extract_rrg
    src/device_reader.cpp
    src/tile_rrg_builder.cpp
    src/coord_to_int.cpp
    src/json_writer.cpp
    src/extract_rrg_main.cpp
    ${CAPNP_SCHEMA_SOURCES}
)

add_executable(extract_net_index
    src/device_reader.cpp
    src/coord_to_int.cpp
    src/phys_netlist_reader.cpp
    src/net_list_writer.cpp
    src/extract_net_index_main.cpp
    schema/interchange/PhysicalNetlist.capnp.c++
    ${CAPNP_SCHEMA_SOURCES}
)

target_include_directories(extract_rrg PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${SCHEMA_DIR}"
    "${CAPNP_INCLUDE}"
)

target_link_libraries(extract_rrg PRIVATE capnp kj ZLIB::ZLIB)

target_include_directories(extract_net_index PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${SCHEMA_DIR}"
    "${CAPNP_INCLUDE}"
)

target_link_libraries(extract_net_index PRIVATE capnp kj ZLIB::ZLIB)

install(TARGETS extract_rrg extract_net_index RUNTIME DESTINATION bin)
