cmake_minimum_required(VERSION 3.25)
project(kamayan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
option(kamayan_BUILD_PYKAMAYAN "Build Python bindings (pyKamayan)" ON)
option(KAMAYAN_ENSURE_MPI4PY
       "Automatically rebuild mpi4py with compatible MPI when mismatch detected"
       OFF)
set(kamayan_NP_TESTING
    "2"
    CACHE STRING "number of cores to use in testing")

# Add cmake module path and find UV early
set(KAMAYAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${KAMAYAN_CMAKE_DIR})
include(FindUV)

# Check UV requirements
if(kamayan_ENABLE_TESTING AND NOT UV_FOUND)
  message(
    FATAL_ERROR
      "kamayan_ENABLE_TESTING is ON but UV was not found.\n"
      "UV is required for testing. Install UV from: https://docs.astral.sh/uv/\n"
      "Or disable testing with: -Dkamayan_ENABLE_TESTING=OFF")
endif()

if(kamayan_BUILD_PYKAMAYAN AND NOT UV_FOUND)
  message(
    WARNING
      "kamayan_BUILD_PYKAMAYAN is ON but UV was not found.\n"
      "UV is required for Python bindings. Disabling pyKamayan build.\n"
      "Install UV from: https://docs.astral.sh/uv/ or disable with -Dkamayan_BUILD_PYKAMAYAN=OFF"
  )
  set(kamayan_BUILD_PYKAMAYAN OFF)
endif()

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

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(
  Python 3.8
  COMPONENTS Interpreter ${DEV_MODULE}
  REQUIRED)

# Detect the installed nanobind package and import it into CMake
# TODO(acreyes): maybe a cmake utility for doing this if we keep adding more...
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind nanobind)
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR)
  find_package(nanobind CONFIG REQUIRED)
endif()

# TODO(acreyes): maybe a cmake utility for doing this if we keep adding more...
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon parthenon)
else()
  find_package(parthenon REQUIRED)
endif()

# Check MPI compatibility with mpi4py for Python bindings This must run after
# Parthenon's MPI discovery
if(ENABLE_MPI AND Python_FOUND)
  include(CheckMPI4PyCompatibility)
  check_mpi_compatibility()
  add_rebuild_mpi4py_target()
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

  # make sure we have a local copy of the baselines Note: UV_FOUND is guaranteed
  # here due to requirement check above
  message(STATUS "Validating baseline files.")
  execute_process(
    COMMAND ${UV_EXECUTABLE} run python -m kamayan.testing.baselines
            validate-tarball RESULT_VARIABLE baseline_result)
  if(NOT baseline_result EQUAL 0)
    message(WARNING "Baseline validation failed")
  endif()

  # regression testing
  add_subdirectory(tests)
endif()

add_subdirectory(src)
add_subdirectory(docs)

# use compdb to generate compile_commands.json at the root level of the repo
# after cmake generates the original one in the build directory. This lets
# clangd get the includes right for header files
if(UV_FOUND)
  add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/compile_commands.json
    COMMAND ${UV_EXECUTABLE} run compdb -p ${CMAKE_BINARY_DIR} list >
            ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json)
  # Add a target to make sure the custom command runs
  add_custom_target(UpdateCompileCommands ALL
                    DEPENDS ${CMAKE_SOURCE_DIR}/compile_commands.json)
else()
  message(
    WARNING
      "UV not found, compile_commands.json will not be updated at project root")
endif()
