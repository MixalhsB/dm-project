# REQUIREMENTS

You will need to have the gcc compiler and a version of Python 3 (3.6 or higher) installed.

Then, run these commands to install Cython:
% sudo python3 -m pip install Cython
% sudo apt-get install cython3

Two further required Python modules, which might need to be installed, are:
% sudo python3 -m pip install numpy
% sudo python3 -m pip install kmedoids


# COMPILATION

From within the "src" directory, in order to compile the "recommend.pyx" file, you need to run:
% cython3 --embed recommend.pyx -o recommend.c
% gcc -Os -w -I <PYTHON_INCLUDE_DIR> -I <NUMPY_INCLUDE_DIR> recommend.c -lpython<VERSION> -o ../bin/recommend

In the second command, replace <PYTHON_INCLUDE_DIR> with the directory you get printed out when running ...
% python3
% >>> from sysconfig import get_paths
% >>> get_paths()['include']
% /usr/include/python3.8        (for example)

... and replace <NUMPY_INCLUDE_DIR> with the directory you get printed out when running ...
% python3
% >>> import numpy
% >>> numpy.get_include()
% /home/michael/.local/lib/python3.8/site-packages/numpy/core/include       (for example)

... and lastly, <VERSION> should be the Python version you are running, for example: 3.8


# EXECUTION

To run the program for a specific patient-and-condition query, simply type:
% ../bin/recommend <DATASET_JSON> <PATIENT_ID> <PATIENT_CONDITION_ID>
As an example:
% ../bin/recommend ../data/datasetB.json 6 pc32

In order to enter the evaluation mode, type the following instead:
% ../bin/recommend <DATASET_JSON> --eval
For instance:
% ../bin/recommend ../data/datasetA.json --eval
