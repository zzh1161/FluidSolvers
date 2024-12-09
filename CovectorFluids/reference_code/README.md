## Acknowledgements
This codebase is forked and modified from the BIMOCQ paper's codebase.

## Compile and Run
To build this project, you will need tbb, openvdb, boost, and CUDA as dependencies.
```
sudo apt-get install libtbb-dev libopenvdb-dev libboost-all-dev
```
This code is built and tested on Ubuntu 16.04 and 18.04, Windows 10, and Mac OS (for 2d portion of the code only) and tested with SM_6X GPUs and above.

2D code and 3D code are separated, you can execute them separately. 
If you do not have an NVIDIA GPU you will not be able to run the 3D portion of the code, but 2D code is still runnable.
For that, simply remove CUDA dependency from the make file (line 7 of file CMakeLists.txt file).

For 2D code, you should specify the simulation method as the second input parameter. 0-7 are Stable Fluids (SF), Reflection (SF+R), Stable and Circulation Preserving Fluids (SCPF), MacCormack (MC), MC+R, BiMocq, Covector Fluids (CF), CF+MCM methods, respectively.
The third parameter is the example case you wish to run, 0-5 represents Taylor Vortices, Leapfrogging Pairs, Ink drop, Ink drop SIGGRAPH Logo, Covector transport with Zalesak's disc, and von Karman Vortex Street, respectively.
```
mkdir build
cd build && cmake ..
make
./Covector2D sim_method sim_setup
```
For 3D experiments, the setup is similar to 2D.
For 2D code, you should specify the simulation method as the second input parameter. 0-7 are Stable Fluids (SF), Reflection (SF+R), Stable and Circulation Preserving Fluids (SCPF), MacCormack (MC), MC+R, BiMocq, Covector Fluids (CF), CF+MCM methods, respectively.
The third parameter is the example case you wish to run, 0-7 represents Trefoil Knot, Leapfrogging Rings, Smoke Plume, Pyroclastic Cloud, Ink Jet, Delta Wing, and Bunny Meteor, respectively.
```
./Covector3D sim_method sim_setup
```
Generated .vdb file will be located in COVECTOR/Out folder, you can use any software(e.g. Houdini) to visualize.
