#!/bin/bash

python3 setup.py build_ext --inplace
cp event.*so ../.
cp weighter.*so ../.
