find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_DETECTION_ESTIMATION gnuradio-detection_estimation)

FIND_PATH(
    GR_DETECTION_ESTIMATION_INCLUDE_DIRS
    NAMES gnuradio/detection_estimation/api.h
    HINTS $ENV{DETECTION_ESTIMATION_DIR}/include
        ${PC_DETECTION_ESTIMATION_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_DETECTION_ESTIMATION_LIBRARIES
    NAMES gnuradio-detection_estimation
    HINTS $ENV{DETECTION_ESTIMATION_DIR}/lib
        ${PC_DETECTION_ESTIMATION_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-detection_estimationTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_DETECTION_ESTIMATION DEFAULT_MSG GR_DETECTION_ESTIMATION_LIBRARIES GR_DETECTION_ESTIMATION_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_DETECTION_ESTIMATION_LIBRARIES GR_DETECTION_ESTIMATION_INCLUDE_DIRS)
