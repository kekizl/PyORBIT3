#include "orbit_mpi.hh"

#include "wrap_utils.hh"
#include "wrap_matrix.hh"
#include "wrap_phase_vector.hh"
#include "wrap_py_base_field_source.hh"
#include "wrap_field_source_container.hh"
#include "wrap_function.hh"
#include "wrap_splinech.hh"
#include "wrap_statmoments2d.hh"
#include "wrap_bunch_extrema_calculator.hh"
#include "wrap_gauss_legendre_integrator.hh"
#include "wrap_polynomial.hh"

namespace wrap_orbit_utils{

  void error(const char* msg){ ORBIT_MPI_Finalize(msg); }
	
  static PyMethodDef UtilsModuleMethods[] = { {NULL,NULL} };

#ifdef __cplusplus
extern "C" {
#endif

  void initutils(){
    //create new module
    PyObject* module = Py_InitModule("orbit_utils",UtilsModuleMethods);		
		//add the other classes init
		wrap_utils_martix::initMatrix(module);
		wrap_utils_phase_vector::initPhaseVector(module);
		wrap_utils_py_base_field_source::initPyBaseFieldSource(module);
		wrap_field_source_container::initFieldSourceContainer(module);
		wrap_function::initFunction(module);
		wrap_splinech::initSplineCH(module);
		wrap_statmoments2d::initstatmoments2d(module);
		wrap_utils_bunch::initBunchExtremaCalculator(module);
		wrap_gl_integrator::initGLIntegrator(module);
		wrap_polynomial::initPolynomial(module);		
  }

	PyObject* getOrbitUtilsType(const char* name){
		PyObject* mod = PyImport_ImportModule("orbit_utils");
		PyObject* pyType = PyObject_GetAttrString(mod,name);
		Py_DECREF(mod);
		Py_DECREF(pyType);
		return pyType;
	}
	
	
#ifdef __cplusplus
}
#endif

//end of namespace wrap_orbit_utils
}

