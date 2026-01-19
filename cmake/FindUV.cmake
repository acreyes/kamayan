# cmake/FindUV.cmake
# Find the UV Python package manager and provide utilities
#
# Provides:
#   UV_EXECUTABLE - Path to uv binary (cached, user-configurable)
#   UV_FOUND      - Boolean, TRUE if uv found and working
#   UV_VERSION    - Version string (e.g., "0.5.4")
#   uv_command()  - Wrapper function for ${UV_EXECUTABLE} run <cmd>

# Allow user override via -DUV_EXECUTABLE=/path/to/uv
if(NOT UV_EXECUTABLE)
  find_program(UV_EXECUTABLE NAMES uv)
endif()

if(UV_EXECUTABLE)
  # Verify it works and get version
  execute_process(
    COMMAND ${UV_EXECUTABLE} --version
    OUTPUT_VARIABLE UV_VERSION_OUTPUT
    ERROR_VARIABLE UV_VERSION_ERROR
    RESULT_VARIABLE UV_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  
  if(UV_VERSION_RESULT EQUAL 0)
    # Parse version (format: "uv 0.5.4" or "uv 0.5.4 (hash)")
    if(UV_VERSION_OUTPUT MATCHES "uv ([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(UV_VERSION "${CMAKE_MATCH_1}" CACHE STRING "UV version")
    else()
      set(UV_VERSION "unknown" CACHE STRING "UV version")
    endif()
    
    set(UV_FOUND TRUE CACHE BOOL "UV found and working")
    message(STATUS "Found UV: ${UV_EXECUTABLE} (version ${UV_VERSION})")
  else()
    set(UV_FOUND FALSE CACHE BOOL "UV found and working")
    set(UV_EXECUTABLE "" CACHE FILEPATH "Path to UV executable" FORCE)
    message(STATUS "Found UV executable but it failed to run: ${UV_EXECUTABLE}")
  endif()
else()
  set(UV_FOUND FALSE CACHE BOOL "UV found and working")
  message(STATUS "UV package manager not found (optional)")
endif()

# Mark as advanced (hide from cmake-gui by default)
mark_as_advanced(UV_EXECUTABLE UV_VERSION UV_FOUND)

# Wrapper function: uv_command()
# Executes: ${UV_EXECUTABLE} run <command...>
#
# Required:
#   COMMAND <cmd> [args...]
#
# Optional:
#   OUTPUT_VARIABLE <var>
#   ERROR_VARIABLE <var>
#   RESULT_VARIABLE <var>
#   WORKING_DIRECTORY <dir>
#   OUTPUT_QUIET
#   ERROR_QUIET
#   OUTPUT_STRIP_TRAILING_WHITESPACE
#   ERROR_STRIP_TRAILING_WHITESPACE
#
# Example:
#   uv_command(
#     COMMAND python -m mymodule --flag
#     OUTPUT_VARIABLE output
#     RESULT_VARIABLE result
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#   )
#
function(uv_command)
  if(NOT UV_FOUND)
    message(FATAL_ERROR 
      "uv_command() called but UV is not available.\n"
      "Ensure UV_FOUND is TRUE before calling uv_command().")
  endif()
  
  # Parse arguments
  cmake_parse_arguments(
    UV_CMD                                      # prefix
    "OUTPUT_QUIET;ERROR_QUIET;OUTPUT_STRIP_TRAILING_WHITESPACE;ERROR_STRIP_TRAILING_WHITESPACE"  # options
    "OUTPUT_VARIABLE;ERROR_VARIABLE;RESULT_VARIABLE;WORKING_DIRECTORY"  # one-value keywords
    "COMMAND"                                   # multi-value keywords
    ${ARGN}
  )
  
  if(NOT UV_CMD_COMMAND)
    message(FATAL_ERROR "uv_command() requires COMMAND argument")
  endif()
  
  # Build execute_process arguments
  set(exec_args COMMAND ${UV_EXECUTABLE} run ${UV_CMD_COMMAND})
  
  # Add optional arguments
  if(UV_CMD_WORKING_DIRECTORY)
    list(APPEND exec_args WORKING_DIRECTORY ${UV_CMD_WORKING_DIRECTORY})
  endif()
  
  if(UV_CMD_OUTPUT_QUIET)
    list(APPEND exec_args OUTPUT_QUIET)
  else()
    list(APPEND exec_args OUTPUT_VARIABLE _output)
    if(UV_CMD_OUTPUT_STRIP_TRAILING_WHITESPACE)
      list(APPEND exec_args OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif()
  
  if(UV_CMD_ERROR_QUIET)
    list(APPEND exec_args ERROR_QUIET)
  else()
    list(APPEND exec_args ERROR_VARIABLE _error)
    if(UV_CMD_ERROR_STRIP_TRAILING_WHITESPACE)
      list(APPEND exec_args ERROR_STRIP_TRAILING_WHITESPACE)
    endif()
  endif()
  
  list(APPEND exec_args RESULT_VARIABLE _result)
  
  # Execute
  execute_process(${exec_args})
  
  # Return via parent scope (only if variables were specified)
  if(UV_CMD_OUTPUT_VARIABLE AND NOT UV_CMD_OUTPUT_QUIET)
    set(${UV_CMD_OUTPUT_VARIABLE} "${_output}" PARENT_SCOPE)
  endif()
  
  if(UV_CMD_ERROR_VARIABLE AND NOT UV_CMD_ERROR_QUIET)
    set(${UV_CMD_ERROR_VARIABLE} "${_error}" PARENT_SCOPE)
  endif()
  
  if(UV_CMD_RESULT_VARIABLE)
    set(${UV_CMD_RESULT_VARIABLE} "${_result}" PARENT_SCOPE)
  endif()
endfunction()
