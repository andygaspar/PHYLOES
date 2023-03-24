

# cd src
# rm fast_me_
echo "$PWD"

g++ -c -fopenmp -fPIC bridge.cc fastme.cc SPR.cc bme.cc bNNI.cc graph.cc heap.cc traverse.cc initialiser.cc utils.cc
g++ -shared -Wl,-soname,bridge.so -o bridge.so. bridge.o bme.o bNNI.o graph.o heap.o SPR.o traverse.o utils.o initialiser.o fastme.o -fopenmp -lm -lpthread
echo "$PWD"

#g++ -Wall -fPIC -c *.cc
#g++ -shared -Wl,-soname,libctest.so -o libctest.so.   *.o -fopenmp -lm -lpthread
#g++  -c -fopenmp -fPIC PSO/bridge_.cpp -o PSO/bridge.o
#g++ -shared -fopenmp -Wl,-soname,PSO/bridge.so -o PSO/bridge.so PSO/bridge.o

rm *o


# echo "$PWD"



# cd ..
# ./src/fast_me_ -i mat.mat -m b -n -s -f 17
# ./fastme -i mat.mat -m b -n -s -u init_topology.nwk -f 17

