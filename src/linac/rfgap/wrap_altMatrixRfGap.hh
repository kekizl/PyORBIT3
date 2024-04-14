#ifndef WRAP_ALT_MATRIX_RF_GAP_H
#define WRAP_ALT_MATRIX_RF_GAP_H

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

  namespace wrap_linac{
    void initaltMatrixRfGap(PyObject* module);
  }

#ifdef __cplusplus
}
#endif

#endif
