cmake_minimum_required(VERSION 3.25)
project(kamayan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If the user doesn't specify a build type, prefer Release
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

option(kamayan_ENABLE_TESTING "Enable kamayan test" ON)
option(kamayan_BUILD_DOCS "Build the kamayan docs" OFF)

# -------parthenon----- ##
set(PARTHENON_ENABLE_PYTHON_MODULE_CHECK
    ${kamayan_ENABLE_TESTING}
    CACHE
      BOOL
      "Check if local python version contains all modules required for running tests."
)

set(PARTHENON_ENABLE_TESTING
    OFF
    CACHE BOOL "Disable Parthenon testing.")
set(PARTHENON_DISABLE_OPENMP
    ON
    CACHE BOOL "Disable OpenMP")
set(PARTHENON_DISABLE_EXAMPLES
    ON
    CACHE BOOL "Don't build Parthenon examples.")
set(PARTHENON_DISABLE_SPARSE
    ON
    CACHE BOOL "Disable sparse (not used in kamayan yet)")

# TODO(acreyes): maybe a cmake utility for doing this if we keep adding more...
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon parthenon)
else()
  find_package(parthenon REQUIRED)
endif()

# -------singularity-eos----- ##
set(SINGULARITY_USE_KOKKOS
    ON
    CACHE BOOL "enable kokkos for eos")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos)
else()
  find_package(singularity-eos REQUIRED)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(kamayan_ENABLE_TESTING)
  # unit tests w/ gtest
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(GTEST_FORCE_SHARED_CRT
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(CTest)
  enable_testing()

  # regression testing
  add_subdirectory(tests)
endif()

add_subdirectory(src)
add_subdirectory(docs)

# use compdb to generate compile_commands.json at the root level of the repo
# after cmake generates the original one in the build directory. This lets
# clangd get the includes right for header files
add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/compile_commands.json
  COMMAND uv run compdb -p ${CMAKE_BINARY_DIR} list >
          ${CMAKE_SOURCE_DIR}/compile_commands.json
  DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json)
# Add a target to make sure the custom command runs
add_custom_target(UpdateCompileCommands ALL
                  DEPENDS ${CMAKE_SOURCE_DIR}/compile_commands.json)
