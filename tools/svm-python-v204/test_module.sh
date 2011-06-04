#!/bin/sh

modulename = $1
./svm_python_learn -c 1.0 --m ${modulename} multi-example/train.dat multi-example/model.dat
./svm_python_train -c 1.0 --m ${modulename} multi-example/train.dat multi-example/model.dat
