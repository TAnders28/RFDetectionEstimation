# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT detection_estimation_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT detection_estimation_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND detection_estimation_python_files
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(detection_estimation
   ../../..
   gr::detection_estimation
   "${detection_estimation_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(TARGET detection_estimation_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:detection_estimation_python>
    ${PROJECT_BINARY_DIR}/test_modules/gnuradio/detection_estimation/
)

install(TARGETS detection_estimation_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/detection_estimation COMPONENT pythonapi)
