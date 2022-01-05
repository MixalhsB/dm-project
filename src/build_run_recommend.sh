cython3 --embed recommend.pyx -o recommend.c
gcc -Os -I /usr/include/python3.8 recommend.c -lpython3.8 -o ../bin/recommend
../bin/recommend
