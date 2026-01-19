#!/usr/bin/env python3
"""Detect MPI implementation used by mpi4py.

This script is called by CMake to determine which MPI implementation
mpi4py was built against. It outputs the result in a format that CMake
can easily parse: <implementation>:<version>

Exit codes:
    0: Success (even if mpi4py not found - output indicates status)
"""

import re
import sys


def normalize_mpi_name(mpi_string):
    """Normalize MPI implementation name from version string.

    Args:
        mpi_string: Raw MPI library version string

    Returns:
        Tuple of (normalized_name, version) or (None, None) if unknown
    """
    mpi_lower = mpi_string.lower()

    # Define patterns for different MPI implementations
    patterns = [
        (r"open\s*mpi.*?(\d+\.\d+(?:\.\d+)?)", "openmpi"),
        (r"mpich.*?(?:version[:\s]+)?(\d+\.\d+(?:\.\d+)?)", "mpich"),
        (r"mvapich.*?(\d+\.\d+(?:\.\d+)?)", "mpich"),  # MVAPICH is MPICH-based
        (r"intel.*?mpi.*?(\d+\.\d+(?:\.\d+)?)", "intelmpi"),
        (r"msmpi.*?(\d+\.\d+(?:\.\d+)?)", "msmpi"),
        (r"microsoft\s+mpi.*?(\d+\.\d+(?:\.\d+)?)", "msmpi"),
        (r"cray\s+mpi.*?(\d+\.\d+(?:\.\d+)?)", "craympi"),
        (r"spectrum\s+mpi.*?(\d+\.\d+(?:\.\d+)?)", "spectrummpi"),
    ]

    for pattern, impl_name in patterns:
        match = re.search(pattern, mpi_lower)
        if match:
            version = match.group(1)
            return impl_name, version

    return None, None


def detect_mpi4py_mpi():
    """Detect MPI implementation from mpi4py.

    Returns:
        String in format '<implementation>:<version>' or status string
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return "NOT_FOUND"
    except Exception as e:
        return f"ERROR:Failed to import mpi4py: {e}"

    try:
        version_string = MPI.Get_library_version()
    except Exception as e:
        return f"ERROR:Failed to get MPI library version: {e}"

    # Normalize and extract implementation and version
    impl_name, version = normalize_mpi_name(version_string)

    if impl_name is None:
        # Could not parse - return unknown with raw string (truncated)
        truncated = version_string.split("\n")[0][:100]
        return f"UNKNOWN:{truncated}"

    return f"{impl_name}:{version}"


if __name__ == "__main__":
    result = detect_mpi4py_mpi()
    print(result)
    sys.exit(0)
