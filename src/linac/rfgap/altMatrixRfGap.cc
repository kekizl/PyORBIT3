/**
This class represents a simplified RF gap. It acts on the coordinates like a transport matrix.
There are no nonlinear effects. It should be analog of RF Gap of XAL online model or Trace3D.
For this RF gap we know the E0TL, frequency, and phase only.

The description of the models can be found in
A. Shishlo, J. Holmes,
"Physical Models for Particle Tracking Simulations in the RF Gap",
ORNL Tech. Note ORNL/TM-2015/247, June 2015
*/

#include "altMatrixRfGap.hh"
#include "ParticleMacroSize.hh"

#include "Bunch.hh"
#include "bessel.hh"
#include "OrbitConst.hh"

#include <iostream>
#include <cstring>
#include <cmath>
#include "VectorPerThread_kernel.cuh"

using namespace OrbitUtils;

// Constructor
altMatrixRfGap::altMatrixRfGap(): CppPyWrapper(NULL)
{
}

// Destructor
altMatrixRfGap::~altMatrixRfGap()
{
}

/** Tracks the Bunch trough the RF gap. */
void altMatrixRfGap::trackBunch(Bunch* bunch, double frequency, double E0TL, double phase){
	// E0TL is a maximal energy gain in the gap. It is in GeV.
	// RF frequency is in Hz
	// RF phase in radians
	bunch->compress();
	SyncPart* syncPart = bunch->getSyncPart();
	double gamma = syncPart->getGamma();
	double beta = syncPart->getBeta();
	double mass = bunch->getMass();
	double charge = bunch->getCharge();
	double eKin_in = syncPart->getEnergy();
	double chargeE0TLsin = charge*E0TL*sin(phase);
	double delta_eKin = charge*E0TL*cos(phase);
	//calculate params in the middle of the gap
	syncPart->setMomentum(syncPart->energyToMomentum(eKin_in + delta_eKin/2.0));
	double gamma_gap = syncPart->getGamma();
	double beta_gap = syncPart->getBeta();
	//now move to the end of the gap
	double eKin_out = eKin_in + delta_eKin;
	syncPart->setMomentum(syncPart->energyToMomentum(eKin_out));
	//the base RF gap is simple - no phase correction. The delta time in seconds
	double delta_time = 0.;
	syncPart->setTime(syncPart->getTime() + delta_time);
	double gamma_out = syncPart->getGamma();
	double beta_out = syncPart->getBeta();
	double prime_coeff = (beta*gamma)/(beta_out*gamma_out);
	//wave momentum
	double k = 2.0*OrbitConst::PI*frequency/OrbitConst::c;
	double phase_time_coeff = k/beta;
	//transverse focusing coeff
	double kappa = - charge*E0TL*k/(2.0*mass*beta_gap*beta_gap*beta_out*gamma_gap*gamma_gap*gamma_out);
	double d_rp = kappa*sin(phase);


    //Matrix and vector dimensions
    	const int numVectors = bunch->getSize();	
	const int matrixSize = 6;
	const int vectorSize = 6;
	
	double **coordArray = bunch->coordArr();
	int numElements = numVectors * vectorSize;

	// Create a new double array to hold the combined data
	double *h_vectors = new double[numElements];

	// Copy the data from coordArray to h_vectors
	for (int i = 0; i < numVectors; ++i) {
	    std::memcpy(h_vectors + i * vectorSize, coordArray[i], sizeof(double) * vectorSize);
	}

	double *h_matrix = new double[matrixSize * matrixSize];
	double transfMatrix[6][6] = {
	    {1, 0, 0, 0, 0, 0},
	    {d_rp, prime_coeff, 0, 0, 0, 0},
	    {0, 0, 1, 0, 0, 0},
	    {0, 0, d_rp, prime_coeff, 0, 0},
	    {0, 0, 0, 0, 1, 0},
	    {0, 0, 0, 0, d_rp, prime_coeff}
	};

	// Copy values from transfMatrix to h_matrix
	for (int i = 0; i < matrixSize; ++i) {
	    for (int j = 0; j < matrixSize; ++j) {
		h_matrix[i * matrixSize + j] = transfMatrix[i][j];
	    }
	}
	double *h_results = new double[numVectors * vectorSize];

    double *d_vectors, *d_matrix, *d_results;
    cudaAllocateMemory(&d_vectors, &d_matrix, &d_results, numVectors, vectorSize, matrixSize);

    // Copy host data to device
    cudaSet(h_vectors, h_matrix, d_vectors, d_matrix, numVectors, vectorSize, matrixSize);
    runMatrixVectorMultiplyKernel(d_vectors, d_matrix, d_results, numVectors, vectorSize, matrixSize);

    // Copy results from device to host
    cudaCopyDeviceToHost(d_vectors, d_results, h_results, numVectors, vectorSize);

  
 
    // Free memory
    cudaFreeAll(d_matrix, d_vectors, d_results);

}
