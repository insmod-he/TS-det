#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee train_lenet_bn_dropout_5.log
