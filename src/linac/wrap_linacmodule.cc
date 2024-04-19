#include "orbit_mpi.hh"

#include "wrap_linacmodule.hh"
#include "wrap_BaseRfGap.hh"
#include "wrap_BaseRfGap_slow.hh"
#include "wrap_MatrixRfGap.hh"
#include "wrap_altMatrixRfGap.hh"
#include "wrap_RfGapTTF.hh"
#include "wrap_RfGapTTF_slow.hh"
#include "wrap_SuperFishFieldSource.hh"
#include "wrap_RfGapThreePointTTF.hh"
#include "wrap_RfGapThreePointTTF_slow.hh"
#include "wrap_linac_tracking.hh"

static PyMethodDef linacmoduleMethods[] = { {NULL,NULL} };

static struct PyModuleDef linacModDef =
        {
                PyModuleDef_HEAD_INIT,
                "linac", "Linac C++ classes",
                -1,
                linacmoduleMethods
        };


#ifdef __cplusplus
extern "C" {
#endif
  namespace wrap_linac{

    PyMODINIT_FUNC initlinac(){
        //create new module
        PyObject* module = PyModule_Create(&linacModDef);
        //add the other classes init
        wrap_linac::initBaseRfGap(module);
        wrap_linac::initBaseRfGap_slow(module);
        wrap_linac::initMatrixRfGap(module);
        wrap_linac::initaltMatrixRfGap(module);
	wrap_linac::initRfGapTTF(module);
        wrap_linac::initRfGapTTF_slow(module);
        wrap_linac::initSuperFishFieldSource(module);
        wrap_linac::initRfGapThreePointTTF(module);
        wrap_linac::initRfGapThreePointTTF_slow(module);
        //initialization of the linac tracking module
        wrap_linac_tracking::initlinactracking(module);
        return module;
    }

    PyObject* getLinacType(char* name){
        PyObject* mod = PyImport_ImportModule("_linac");
        PyObject* pyType = PyObject_GetAttrString(mod,name);
        Py_DECREF(mod);
        Py_DECREF(pyType);
        return pyType;
    }
  }

#ifdef __cplusplus
}
#endif
