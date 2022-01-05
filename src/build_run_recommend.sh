cython3 --embed recommend.pyx -o recommend.c
gcc -Os -w -I /usr/include/python3.8 -I /home/michael/.local/lib/python3.8/site-packages/numpy/core/include recommend.c -lpython3.8 -o ../bin/recommend
../bin/recommend
