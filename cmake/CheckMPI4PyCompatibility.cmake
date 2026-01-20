# CheckMPI4PyCompatibility.cmake
#
# This module checks that mpi4py's MPI implementation matches the MPI
# implementation discovered by CMake. This prevents runtime errors when
# using kamayan's Python bindings with MPI.
#
# Provides:
#   check_mpi_compatibility() - Main function to check MPI compatibility
#   rebuild_mpi4py_target()   - CMake target for manual rebuild
#
# Options:
#   kamayan_ENSURE_MPI4PY - Auto-rebuild mpi4py on mismatch (default: OFF)
#   kamayan_VERBOSE_MPI_CHECK - Show detailed detection logic (default: OFF)
#
# Implementation mismatch = FATAL_ERROR
# Version mismatch only = WARNING

option(kamayan_VERBOSE_MPI_CHECK 
  "Show detailed MPI compatibility check information" 
  OFF)

# Helper function to print verbose messages
function(_verbose_message)
  if(kamayan_VERBOSE_MPI_CHECK)
    message(STATUS "[MPI Check] ${ARGN}")
  endif()
endfunction()

# Normalize MPI implementation name to lowercase identifier
function(normalize_mpi_name input_name output_var)
  string(TOLOWER "${input_name}" name_lower)
  
  # Detect different MPI implementations by pattern matching
  if(name_lower MATCHES "open.?mpi")
    set(${output_var} "openmpi" PARENT_SCOPE)
  elseif(name_lower MATCHES "mvapich")
    # MVAPICH is MPICH-based
    set(${output_var} "mpich" PARENT_SCOPE)
  elseif(name_lower MATCHES "mpich")
    set(${output_var} "mpich" PARENT_SCOPE)
  elseif(name_lower MATCHES "intel.*mpi")
    set(${output_var} "intelmpi" PARENT_SCOPE)
  elseif(name_lower MATCHES "msmpi" OR name_lower MATCHES "microsoft.*mpi")
    set(${output_var} "msmpi" PARENT_SCOPE)
  elseif(name_lower MATCHES "cray.*mpi")
    set(${output_var} "craympi" PARENT_SCOPE)
  elseif(name_lower MATCHES "spectrum.*mpi")
    set(${output_var} "spectrummpi" PARENT_SCOPE)
  else()
    # Return as-is if unknown
    set(${output_var} "${name_lower}" PARENT_SCOPE)
  endif()
endfunction()

# Detect CMake's MPI implementation and version
function(detect_cmake_mpi_implementation impl_var version_var)
  _verbose_message("Detecting CMake MPI implementation...")
  
  set(detected_impl "")
  set(detected_version "")
  
  # Strategy 1: Use MPI_CXX_LIBRARY_VERSION_STRING if available
  if(DEFINED MPI_CXX_LIBRARY_VERSION_STRING AND NOT MPI_CXX_LIBRARY_VERSION_STRING STREQUAL "")
    _verbose_message("Found MPI_CXX_LIBRARY_VERSION_STRING: ${MPI_CXX_LIBRARY_VERSION_STRING}")
    set(version_string "${MPI_CXX_LIBRARY_VERSION_STRING}")
    
    # Extract version number
    if(version_string MATCHES "([0-9]+\\.[0-9]+\\.?[0-9]*)")
      set(detected_version "${CMAKE_MATCH_1}")
    endif()
    
    # Normalize implementation name
    normalize_mpi_name("${version_string}" detected_impl)
  endif()
  
  # Strategy 2: Execute mpiexec --version
  if(detected_impl STREQUAL "" AND DEFINED MPIEXEC_EXECUTABLE)
    _verbose_message("Trying MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
    execute_process(
      COMMAND ${MPIEXEC_EXECUTABLE} --version
      OUTPUT_VARIABLE mpiexec_output
      ERROR_VARIABLE mpiexec_error
      RESULT_VARIABLE mpiexec_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    
    if(mpiexec_result EQUAL 0)
      _verbose_message("mpiexec output: ${mpiexec_output}")
      set(version_string "${mpiexec_output}")
      
      # Extract version
      if(version_string MATCHES "([0-9]+\\.[0-9]+\\.?[0-9]*)")
        set(detected_version "${CMAKE_MATCH_1}")
      endif()
      
      normalize_mpi_name("${version_string}" detected_impl)
    endif()
  endif()
  
  # Strategy 3: Execute MPI compiler --version
  if(detected_impl STREQUAL "" AND DEFINED MPI_CXX_COMPILER)
    _verbose_message("Trying MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
    execute_process(
      COMMAND ${MPI_CXX_COMPILER} --version
      OUTPUT_VARIABLE mpicxx_output
      ERROR_VARIABLE mpicxx_error
      RESULT_VARIABLE mpicxx_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    
    if(mpicxx_result EQUAL 0)
      _verbose_message("mpicxx output: ${mpicxx_output}")
      set(version_string "${mpicxx_output}")
      
      # Extract version
      if(version_string MATCHES "([0-9]+\\.[0-9]+\\.?[0-9]*)")
        set(detected_version "${CMAKE_MATCH_1}")
      endif()
      
      normalize_mpi_name("${version_string}" detected_impl)
    endif()
  endif()
  
  # Fallback: Try MPI_C_COMPILER
  if(detected_impl STREQUAL "" AND DEFINED MPI_C_COMPILER)
    _verbose_message("Trying MPI_C_COMPILER: ${MPI_C_COMPILER}")
    execute_process(
      COMMAND ${MPI_C_COMPILER} --version
      OUTPUT_VARIABLE mpicc_output
      ERROR_VARIABLE mpicc_error
      RESULT_VARIABLE mpicc_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    
    if(mpicc_result EQUAL 0)
      _verbose_message("mpicc output: ${mpicc_output}")
      set(version_string "${mpicc_output}")
      
      # Extract version
      if(version_string MATCHES "([0-9]+\\.[0-9]+\\.?[0-9]*)")
        set(detected_version "${CMAKE_MATCH_1}")
      endif()
      
      normalize_mpi_name("${version_string}" detected_impl)
    endif()
  endif()
  
  # Check if detection failed
  if(detected_impl STREQUAL "")
    message(FATAL_ERROR 
      "Failed to detect MPI implementation from CMake.\n"
      "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}\n"
      "MPI_C_COMPILER: ${MPI_C_COMPILER}\n"
      "MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}\n"
      "Please report this issue.")
  endif()
  
  _verbose_message("Detected CMake MPI: ${detected_impl} ${detected_version}")
  
  set(${impl_var} "${detected_impl}" PARENT_SCOPE)
  set(${version_var} "${detected_version}" PARENT_SCOPE)
endfunction()

# Detect mpi4py's MPI implementation and version
function(detect_mpi4py_implementation impl_var version_var)
  _verbose_message("Detecting mpi4py MPI implementation...")
  
  # Execute Python script to detect mpi4py's MPI
  execute_process(
    COMMAND ${Python_EXECUTABLE} ${KAMAYAN_CMAKE_DIR}/detect_mpi4py_mpi.py
    OUTPUT_VARIABLE script_output
    ERROR_VARIABLE script_error
    RESULT_VARIABLE script_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  
  _verbose_message("Script output: ${script_output}")
  
  if(NOT script_result EQUAL 0)
    message(FATAL_ERROR 
      "Failed to run mpi4py detection script.\n"
      "Error: ${script_error}")
  endif()
  
  # Parse output
  if(script_output STREQUAL "NOT_FOUND")
    set(${impl_var} "NOT_FOUND" PARENT_SCOPE)
    set(${version_var} "" PARENT_SCOPE)
    return()
  elseif(script_output MATCHES "^ERROR:(.*)")
    message(FATAL_ERROR 
      "mpi4py detection error: ${CMAKE_MATCH_1}")
  elseif(script_output MATCHES "^UNKNOWN:(.*)")
    message(FATAL_ERROR 
      "Could not parse MPI version string from mpi4py:\n${CMAKE_MATCH_1}\n"
      "Please report this issue.")
  elseif(script_output MATCHES "^([^:]+):([^:]+)$")
    set(${impl_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(${version_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    _verbose_message("Detected mpi4py MPI: ${CMAKE_MATCH_1} ${CMAKE_MATCH_2}")
  else()
    message(FATAL_ERROR 
      "Unexpected output from mpi4py detection script: ${script_output}")
  endif()
endfunction()

# Rebuild mpi4py with correct MPI
function(rebuild_mpi4py)
  message(STATUS "MPI implementation mismatch detected")
  
  set(marker_file "${CMAKE_BINARY_DIR}/.cmake_mpi4py_rebuilt")
  
  # Check if marker file exists and is valid
  if(EXISTS "${marker_file}")
    _verbose_message("Found existing marker file: ${marker_file}")
    file(STRINGS "${marker_file}" marker_contents)
    
    set(marker_mpi_impl "")
    set(marker_mpicc "")
    
    foreach(line ${marker_contents})
      if(line MATCHES "^MPI_IMPLEMENTATION=(.*)$")
        set(marker_mpi_impl "${CMAKE_MATCH_1}")
      elseif(line MATCHES "^MPICC=(.*)$")
        set(marker_mpicc "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    
    # Check if marker is still valid (same MPI)
    if(marker_mpi_impl STREQUAL CMAKE_MPI_IMPLEMENTATION AND 
       marker_mpicc STREQUAL MPI_C_COMPILER)
      message(STATUS "mpi4py already rebuilt with ${CMAKE_MPI_IMPLEMENTATION}, skipping")
      return()
    else()
      _verbose_message("Marker is stale (MPI changed), will rebuild")
      file(REMOVE "${marker_file}")
    endif()
  endif()
  
  # Check if uv is available
  if(NOT UV_FOUND)
    message(FATAL_ERROR 
      "\n"
      "Cannot rebuild mpi4py: 'uv' command not found.\n"
      "\n"
      "Please install uv or manually rebuild mpi4py:\n"
      "  $ uv cache clean mpi4py\n"
      "  $ MPICC=${MPI_C_COMPILER} MPICXX=${MPI_CXX_COMPILER} uv sync --reinstall-package mpi4py\n")
  endif()
  
  message(STATUS "Rebuilding mpi4py with ${CMAKE_MPI_IMPLEMENTATION} ${CMAKE_MPI_VERSION}...")
  
  # Step 1: Clean mpi4py cache
  message(STATUS "Cleaning mpi4py cache...")
  execute_process(
    COMMAND ${UV_EXECUTABLE} cache clean mpi4py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE clean_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  
  # Step 2: Rebuild mpi4py with correct MPI compilers
  message(STATUS "Installing mpi4py from source with MPICC=${MPI_C_COMPILER}...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
            --unset=CC
            --unset=CXX
            MPICC=${MPI_C_COMPILER}
            MPICXX=${MPI_CXX_COMPILER}
            ${UV_EXECUTABLE} sync --reinstall-package mpi4py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE rebuild_result
    OUTPUT_VARIABLE rebuild_output
    ERROR_VARIABLE rebuild_error
  )
  
  if(NOT rebuild_result EQUAL 0)
    message(FATAL_ERROR 
      "Failed to rebuild mpi4py.\n"
      "Output: ${rebuild_output}\n"
      "Error: ${rebuild_error}")
  endif()
  
  # Step 3: Re-check compatibility
  message(STATUS "Re-checking mpi4py compatibility...")
  detect_mpi4py_implementation(new_mpi4py_impl new_mpi4py_version)
  
  if(NOT new_mpi4py_impl STREQUAL CMAKE_MPI_IMPLEMENTATION)
    message(FATAL_ERROR 
      "Rebuild failed: mpi4py still has wrong MPI implementation.\n"
      "Expected: ${CMAKE_MPI_IMPLEMENTATION}\n"
      "Got: ${new_mpi4py_impl}\n"
      "This indicates a deeper issue with the build environment.")
  endif()
  
  # Step 4: Create marker file
  string(TIMESTAMP current_time UTC)
  file(WRITE "${marker_file}"
    "# Auto-generated by CheckMPI4PyCompatibility.cmake\n"
    "TIMESTAMP=${current_time}\n"
    "MPI_IMPLEMENTATION=${CMAKE_MPI_IMPLEMENTATION}\n"
    "MPI_VERSION=${CMAKE_MPI_VERSION}\n"
    "MPICC=${MPI_C_COMPILER}\n"
    "MPICXX=${MPI_CXX_COMPILER}\n"
  )
  
  message(STATUS "Successfully rebuilt mpi4py with ${CMAKE_MPI_IMPLEMENTATION} ${CMAKE_MPI_VERSION}")
endfunction()

# Main compatibility check function
function(check_mpi_compatibility)
  message(STATUS "MPI compatibility check...")
  
  # Detect CMake's MPI
  detect_cmake_mpi_implementation(CMAKE_MPI_IMPLEMENTATION CMAKE_MPI_VERSION)
  set(CMAKE_MPI_IMPLEMENTATION "${CMAKE_MPI_IMPLEMENTATION}" PARENT_SCOPE)
  set(CMAKE_MPI_VERSION "${CMAKE_MPI_VERSION}" PARENT_SCOPE)
  
  # Detect mpi4py's MPI
  detect_mpi4py_implementation(MPI4PY_IMPLEMENTATION MPI4PY_VERSION)
  
  # Print detected implementations
  message(STATUS "CMake MPI:   ${CMAKE_MPI_IMPLEMENTATION} ${CMAKE_MPI_VERSION}")
  message(STATUS "mpi4py MPI:  ${MPI4PY_IMPLEMENTATION} ${MPI4PY_VERSION}")
  
  # Handle mpi4py not found case
  if(MPI4PY_IMPLEMENTATION STREQUAL "NOT_FOUND")
    message(STATUS "mpi4py not yet installed, skipping compatibility check")
    message(STATUS "NOTE: Run 'uv sync' to install Python dependencies")
    return()
  endif()
  
  # Compare implementations (case-insensitive)
  string(TOLOWER "${CMAKE_MPI_IMPLEMENTATION}" cmake_mpi_lower)
  string(TOLOWER "${MPI4PY_IMPLEMENTATION}" mpi4py_lower)
  
  if(NOT cmake_mpi_lower STREQUAL mpi4py_lower)
    # Implementation mismatch - this is fatal
    if(kamayan_ENSURE_MPI4PY)
      rebuild_mpi4py()
      message(STATUS "MPI compatibility check: PASSED ✓")
    else()
      message(FATAL_ERROR
        "\n"
        "═══════════════════════════════════════════════════════════════════\n"
        "MPI IMPLEMENTATION MISMATCH DETECTED\n"
        "═══════════════════════════════════════════════════════════════════\n"
        "\n"
        "CMake discovered MPI: ${CMAKE_MPI_IMPLEMENTATION} ${CMAKE_MPI_VERSION}\n"
        "mpi4py is built with:  ${MPI4PY_IMPLEMENTATION} ${MPI4PY_VERSION}\n"
        "\n"
        "These MPI implementations are incompatible and will cause runtime\n"
        "errors when using kamayan's Python bindings.\n"
        "\n"
        "───────────────────────────────────────────────────────────────────\n"
        "Solutions:\n"
        "───────────────────────────────────────────────────────────────────\n"
        "\n"
        "Option 1: Manually rebuild mpi4py with the correct MPI\n"
        "  $ uv cache clean mpi4py\n"
        "  $ MPICC=${MPI_C_COMPILER} MPICXX=${MPI_CXX_COMPILER} uv sync --reinstall-package mpi4py\n"
        "\n"
        "Option 2: Enable automatic rebuild in CMake\n"
        "  $ cmake -B build -Dkamayan_ENSURE_MPI4PY=ON\n"
        "\n"
        "═══════════════════════════════════════════════════════════════════\n"
      )
    endif()
  elseif(NOT CMAKE_MPI_VERSION STREQUAL MPI4PY_VERSION)
    # Same implementation, different version - warning only
    message(WARNING
      "\n"
      "MPI version mismatch detected (same implementation, different versions).\n"
      "\n"
      "CMake MPI:   ${CMAKE_MPI_IMPLEMENTATION} ${CMAKE_MPI_VERSION}\n"
      "mpi4py MPI:  ${MPI4PY_IMPLEMENTATION} ${MPI4PY_VERSION}\n"
      "\n"
      "This may work but could cause subtle compatibility issues.\n"
      "\n"
      "To rebuild mpi4py with the exact MPI version, run:\n"
      "  cmake -B build -Dkamayan_ENSURE_MPI4PY=ON\n"
    )
    message(STATUS "MPI compatibility check: PASSED (with version mismatch warning)")
  else()
    # Perfect match
    message(STATUS "MPI compatibility check: PASSED ✓")
  endif()
endfunction()

# Add a custom target for manual mpi4py rebuild
function(add_rebuild_mpi4py_target)
  # Check if uv is available before adding target
  if(NOT UV_FOUND)
    return()
  endif()
  
  add_custom_target(rebuild-mpi4py
    COMMAND ${CMAKE_COMMAND} -E echo "Rebuilding mpi4py with ${CMAKE_MPI_IMPLEMENTATION}..."
    COMMAND ${CMAKE_COMMAND} -E env ${UV_EXECUTABLE} cache clean mpi4py
    COMMAND ${CMAKE_COMMAND} -E env
            --unset=CC
            --unset=CXX
            MPICC=${MPI_C_COMPILER}
            MPICXX=${MPI_CXX_COMPILER}
            ${UV_EXECUTABLE} sync --reinstall-package mpi4py
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/.cmake_mpi4py_rebuilt
    COMMAND ${CMAKE_COMMAND} -E echo "Done! Re-run cmake to verify."
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Rebuilding mpi4py with correct MPI implementation"
    VERBATIM
  )
  
  message(STATUS "Added 'rebuild-mpi4py' target for manual mpi4py rebuild")
endfunction()
