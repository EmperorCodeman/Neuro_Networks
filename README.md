# neuro-networks
Open this for formating. Not taking time to learn mark up. Open and spacing should work  

    Run numpy on the gpu for much faster performance. Using CuPy
CuPy can use the nividia driver: CUDA. Install the driver and other dependecies. Other gpu drivers also work. Example amd 
  CuPy: https://cupy.dev/
Use this command to get the nividia CUDA version. CUDA is a driver api for developers to use the gpu 
  nvcc --version
  
Install vs studio. Its c++ compiler is needed 
    add the path to the vs studios c++ compiler in PATH enviromental var windows. Find cll.exe with system wide search  
        C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx86\x64
run these commands to install 
 python -m pip install -U setuptools pip
 #for cuda version 11.2 or later (x86_64) use 
    pip install cupy-cuda11x
 python -m cupyx.tools.install_library --cuda 11.x --library cutensor


 
