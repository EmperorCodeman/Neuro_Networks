# neuro-networks
  
    Run numpy on the gpu for much faster performance. Using CuPy
CuPy can use the nividia driver: CUDA. Install the driver and other dependecies. Other gpu drivers also work. Example amd 
  CuPy: https://cupy.dev/
Use this command to get the nividia CUDA version. CUDA is a driver api for developers to use the gpu 
  nvcc --version
  
 run these commands to install 
 python -m pip install -U setuptools pip
 #for cuda version 11.2 or later (x86_64) use 
    pip install cupy-cuda11x
 python -m cupyx.tools.install_library --cuda 11.x --library cutensor

 
 
