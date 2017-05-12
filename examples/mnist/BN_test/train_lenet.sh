#!/usr/bin/env sh

../../../build/tools/caffe train --solver=./lenet_solver.prototxt --gpu 0 2>&1 | tee MLP_BN.log
