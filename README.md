# voc-demo
    mkdir build
    cd build
    cmake ../
    build
## install dbow2
    cd dbow2
    mkdir build && build
    cmake ../
    make
## opencv not found 
### install opencv
* down load opencv from source 
	https://github.com/opencv/opencv/releases
* install dependence
	sudo apt-get install build-essential
	sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
* build opencv
	cd ~/opencv-
	mkdir build && cd build
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
	make -j7
	sudo make install
