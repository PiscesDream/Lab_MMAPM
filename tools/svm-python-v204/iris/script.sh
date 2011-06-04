#!/bin/sh

#model=multiclass
#model=svmstruct

model=mymulti
shared_options="--m ${model}"
workdir=../../../data
#workdir=../multi-example/
#workdir=./data
train_file=${workdir}/train.dat
model_file=${workdir}/model.dat
test_file=${workdir}/test.dat
output_file=${workdir}/output.dat

echo "Training ..."
$SSVMPATH/svm_python_learn ${shared_options} $* ${train_file} ${model_file}
echo "..."
echo "..."
echo "..."
echo "Testing ..."
$SSVMPATH/svm_python_classify ${shared_options} ${test_file} ${model_file} ${output_file} 
