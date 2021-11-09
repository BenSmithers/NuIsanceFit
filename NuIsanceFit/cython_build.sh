#!/bin/bash

python3 setup.py build_ext --inplace
cp build/lib.linux-x86_64-3.8/NuIsanceFit/* .
rm weighter.c
rm weighter.cpp
rm event.c
rm event.cpp
