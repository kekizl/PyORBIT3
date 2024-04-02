# PyOrbit3 package installation

## 0. Required software

One needs compilers and python development packages, depending on  Linux flavor the package can be called **python-dev** or **python-devel**.

This guide was tested on following configurations

| CPU           | Architecture | OS           | Python  | Compiler     |
|---------------|--------------|--------------|---------|--------------|
| Intel i7-7700 | x86_64       | RHEL 8.7     | 3.9.13  | gcc-8.5      |
|               | x86_64       | Arch         | 3.10.10 | gcc-12.2.1   |
| Apple M2      | arm64        | macOS 13.3.1 | 3.9.6   | clang-14.0.3 |


## 1. Installation from source

First step is to clone the source code:

```bash
git clone https://github.com/PyORBIT-Collaboration/PyORBIT3.git
```

Make sure you have a C++ compiler.

On Debian based distributions:
```
sudo apt-get update
sudo apt-get install build-essential
```

On RedHat based distributions
```
sudo yum update
sudo yum group install "Development Tools"
```

You will also need to install the relevant packages in order to use PyOrbit. You can either do this through conda, or by installing the packages with your preferred package manager.

### Conda Setup (Recommended):

First of all make sure you have conda installed. Then run the following:

```bash
cd pyorbit3
conda env create -n pyorbit --file environment.yml
conda activate pyorbit
```

### Manual Setup:

Make sure that you have the correct python version installed. We require python=3.10. Further the following packages are required, use your preferred packet manager to install them:

- FFTW
- Matplotlib
- Numpy & Scipy
- Mpich

On Debian based distributions:
```
sudo apt-get install python-dev libmpich-dev mpich  zlib1g-dev libfftw3-dev
```

On RedHat based distributions
```
sudo yum install python-devel mpich mpich-devel zlib-devel fftw-devel
```

Then install the Python dependencies using PIP:
```
pip install numpy scipy matplotlib
```

For both of these methods these are the minimum requirements you need to run the examples. Additional packages will be required if you would like to modify and deploy to GitHub. These packages are pre-commit, flake8 and pytest to name a few.

## 2. Build

After you have installed everything, the next step is to build. In order to build the project, navigate to the root pyorbit directory and run the following:

```bash
python setup.py clean
pip install .
```

You need only build the project after a change is made to the core c++ or python classes.

## 3. Run SNS linac example

Navigate to your examples directory:

```bash
cd examples/SNS_Linac/pyorbit3_linac_model/
python pyorbit3_sns_linac_mebt_hebt2.py
```

Additionally if you would like to run the example on multiple MPI nodes you can use the following:

```bash
mpirun -n 4 python pyorbit3_sns_linac_mebt_hebt2.py
```

In the above line you can change the number 4 for however many MPI nodes you would like to test it on.

# Structure
**./src**		- source code for the core ORBIT C++ classes, including
		  wrappers, etc.

**./py**		- python modules and wrapper classes for the core ORBIT
		  classes.

**./ext**		- source code for external modules. Compilations of this
		  code should be placed into **./lib**.

**./lib**  	- .so shared libraries to be used under pyORBIT interpreter.

**./examples**		- pyORBIT3 examples.

**./tests**		- pytests written for the CI Pipeline.

### CUDA Submodule (Beta)

This project includes a submodule for CUDA, a parallel computing platform and programming model developed by NVIDIA. CUDA enables developers to leverage the power of NVIDIA GPUs for parallel processing tasks, making it particularly useful for computationally intensive applications such as machine learning, scientific simulations, and image processing.

#### Submodule Description

The CUDA submodule contains kernels and wrappers to leverage the power of GPU-accelerated computing. It includes headers, runtime libraries, and utilities provided by NVIDIA to facilitate the development and execution of CUDA applications.

### Installing CUDA and Drivers
### Installing CUDA and Drivers

1. **Check GPU Compatibility:**
   - Verify that your GPU is compatible with the CUDA version you intend to install. Check the CUDA documentation for compatibility information.

2. **Install NVIDIA Drivers:**
   - Install the appropriate NVIDIA drivers for your GPU. You can use the `ubuntu-drivers` tool to identify and install the recommended drivers:
     ```bash
     sudo ubuntu-drivers autoinstall
     ```

3. **Download CUDA Toolkit:**
   - Download the CUDA Toolkit from the NVIDIA website. Choose the version that matches your system configuration and requirements.

4. **Run Installer:**
   - Navigate to the directory where the CUDA Toolkit installer is downloaded and run it:
     ```bash
     sudo sh cuda_*.run
     ```
5. **Environment Setup:**
   - Update your system's PATH environment variable to include CUDA binaries. Add the following lines to your `.bashrc` or `.bash_profile` file:
     ```bash
     export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
     ```

6. **Verify Installation:**
   - After installation, verify that CUDA is correctly installed by checking the CUDA version:
     ```bash
     nvcc --version
     ```

#### Usage

To use the CUDA submodule in this project, follow these steps:

1. **CUDA**

After installing cuda build the project anew
```bash
python setup.py clean --all
python setup.py build
pip install . 
```

Navigate to the GPU Examples directory and choose one of the examples
