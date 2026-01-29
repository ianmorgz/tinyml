#pragma once
#include <stdexcept>

// exceptions enabled
#ifndef TINYML_ENABLE_EXCEPTIONS
#define TINYML_ENABLE_EXCEPTIONS 1
#endif

#if TINYML_ENABLE_EXCEPTIONS
    #define TINYML_EXCEPTION(msg) throw std::runtime_error(msg)
#else
    #include <cstdlib>
    #define TINYML_EXCEPTION(msg) std::abort()
#endif
