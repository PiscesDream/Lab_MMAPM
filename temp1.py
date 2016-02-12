Using gpu device 1: GeForce GTX TITAN X
/home/shaofan/.local/lib/python2.7/site-packages/theano/scan_module/scan_perform_ext.py:133: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
  from scan_perform.scan_perform import *
BIT
(2720, 600)
(2720,)
(12.34375, array([[ 100.,   19.,    2.,    1.,    0.,    0.,    0.,    0.,    0.,    0.],
       [  87.,   18.,   13.,    7.,    0.,    0.,    0.,    0.,    0.,    0.],
       [  41.,   54.,   14.,   25.,    0.,    0.,    0.,    0.,    0.,    0.],
       [  21.,   32.,   44.,    7.,   16.,    1.,    1.,    0.,    0.,    0.],
       [  10.,   21.,   44.,   38.,    6.,   15.,    7.,    1.,    0.,    0.],
       [   5.,   11.,   49.,   26.,   23.,    0.,   11.,    3.,    1.,    1.],
       [   1.,   10.,   32.,   37.,   18.,   15.,    2.,   10.,    4.,    5.],
       [   1.,    4.,   22.,   30.,   27.,   10.,    7.,    2.,   17.,   11.],
       [   1.,    3.,   15.,   15.,   23.,   13.,    8.,   17.,    3.,   28.],
       [   0.,    2.,   13.,   22.,   20.,   12.,    6.,   17.,   16.,    6.]]))
Before pca: x.shape: (2720, 600)
	Explained variance ratio 0.942611537816
shape: (2720, 100)
train-acc: 35.074%                                
train confusion matrix:
 [[ 241.   34.    3.    0.    0.    0.    0.    0.    0.    0.]
 [ 112.  153.    9.    1.    0.    0.    0.    0.    0.    0.]
 [  41.   84.  130.   10.    1.    0.    0.    0.    0.    0.]
 [  12.   46.   95.  113.   10.    2.    0.    0.    0.    0.]
 [   6.    8.   58.  117.   66.    3.    0.    0.    0.    0.]
 [   2.    5.   20.   78.   97.   67.    1.    0.    0.    0.]
 [   0.    0.    9.   34.   61.   89.   71.    2.    0.    0.]
 [   0.    1.    1.   12.   22.   51.   92.   87.    0.    3.]
 [   0.    0.    1.    1.   11.   31.   83.  137.    4.    6.]
 [   0.    0.    0.    6.    6.   30.   81.  139.    2.   22.]]
test-acc: 10.391%
test confusion matrix:
 [[ 82.  37.   2.   1.   0.   0.   0.   0.   0.   0.]
 [ 70.  28.  20.   7.   0.   0.   0.   0.   0.   0.]
 [ 25.  63.  16.  29.   1.   0.   0.   0.   0.   0.]
 [ 10.  23.  62.   3.  20.   3.   1.   0.   0.   0.]
 [  5.  13.  46.  57.   1.  15.   4.   1.   0.   0.]
 [  2.   3.  26.  52.  25.   1.  20.   1.   0.   0.]
 [  0.   3.   7.  32.  31.  46.   0.  11.   0.   4.]
 [  0.   0.   3.  11.  28.  36.  48.   1.   1.   3.]
 [  0.   0.   2.   8.  14.  31.  44.  11.   0.  16.]
 [  0.   0.   3.   3.  15.  33.  44.  13.   2.   1.]]
neighbor cost: 130503888.415
Iter: 0
	Violated triples: 8980/10000
	lr: 1e-05
	Share Pull:  65251936.0
	Share Push:  805972.116089
Iter: 1
	Violated triples: 8980/10000
	lr: 1.01e-05
	Share Pull:  3170776.25
	Share Push:  6349369.74958
Iter: 2
	Violated triples: 8980/10000
	lr: 1.0201e-05
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 3
	Violated triples: 8980/10000
	lr: 7.65075e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 4
	Violated triples: 8980/10000
	lr: 5.7380625e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 5
	Violated triples: 8980/10000
	lr: 4.303546875e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 6
	Violated triples: 8980/10000
	lr: 3.22766015625e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 7
	Violated triples: 8980/10000
	lr: 2.42074511719e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 8
	Violated triples: 8980/10000
	lr: 1.81555883789e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 9
	Violated triples: 8980/10000
	lr: 1.36166912842e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 10
	Violated triples: 8980/10000
	lr: 1.02125184631e-06
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 11
	Violated triples: 8980/10000
	lr: 7.65938884735e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 12
	Violated triples: 8980/10000
	lr: 5.74454163551e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 13
	Violated triples: 8980/10000
	lr: 4.30840622663e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 14
	Violated triples: 8980/10000
	lr: 3.23130466998e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 15
	Violated triples: 8980/10000
	lr: 2.42347850248e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 16
	Violated triples: 8980/10000
	lr: 1.81760887686e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 17
	Violated triples: 8980/10000
	lr: 1.36320665765e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 18
	Violated triples: 8980/10000
	lr: 1.02240499323e-07
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 19
	Violated triples: 8980/10000
	lr: 7.66803744926e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 20
	Violated triples: 8980/10000
	lr: 5.75102808695e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 21
	Violated triples: 8980/10000
	lr: 4.31327106521e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 22
	Violated triples: 8980/10000
	lr: 3.23495329891e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 23
	Violated triples: 8980/10000
	lr: 2.42621497418e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 24
	Violated triples: 8980/10000
	lr: 1.81966123063e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 25
	Violated triples: 8980/10000
	lr: 1.36474592298e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 26
	Violated triples: 8980/10000
	lr: 1.02355944223e-08
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 27
	Violated triples: 8980/10000
	lr: 7.67669581674e-09
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 28
	Violated triples: 8980/10000
	lr: 5.75752186256e-09
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
Iter: 29
	Violated triples: 8980/10000
	lr: 4.31814139692e-09
	Share Pull:  5282280448.0
	Share Push:  1155458051.67
shape: (2720, 36)
train-acc: 45.956%                                
train confusion matrix:
 [[ 249.   23.    4.    2.    0.    0.    0.    0.    0.    0.]
 [  72.  180.   18.    3.    0.    1.    0.    0.    0.    1.]
 [  36.   59.  153.   14.    1.    3.    0.    0.    0.    0.]
 [  13.   27.   68.  156.   11.    3.    0.    0.    0.    0.]
 [   8.   11.   33.   82.  114.    6.    2.    0.    2.    0.]
 [   4.    9.   20.   64.   71.   94.    6.    1.    1.    0.]
 [   0.    6.    6.   26.   49.   77.   94.    5.    0.    3.]
 [   1.    0.    5.    7.   20.   37.   75.  114.    2.    8.]
 [   0.    3.    3.    4.   10.   16.   50.  127.   34.   27.]
 [   0.    0.    3.    6.    3.   16.   55.  119.   22.   62.]]
test-acc: 17.031%
test confusion matrix:
 [[ 86.  25.   6.   4.   0.   1.   0.   0.   0.   0.]
 [ 51.  41.  23.   5.   2.   2.   0.   1.   0.   0.]
 [ 21.  40.  30.  32.   5.   4.   1.   1.   0.   0.]
 [  4.  12.  57.  11.  24.   8.   4.   2.   0.   0.]
 [  2.  18.  28.  55.   7.  23.   5.   3.   0.   1.]
 [  0.   7.  20.  32.  36.   7.  21.   4.   2.   1.]
 [  2.   1.   8.  26.  27.  36.  15.  14.   2.   3.]
 [  1.   0.   9.   5.  20.  20.  39.  15.  11.  11.]
 [  0.   0.   4.   5.   8.  14.  31.  31.   2.  31.]
 [  0.   1.   4.   4.  15.  11.  28.  29.  18.   4.]]
neighbor cost: 5548139047.62
Iter: 0
	Violated triples: 8971/10000
	lr: 3.23860604769e-09
	Share Pull:  2774067200.0
	Share Push:  99864284.8913
Iter: 1
	Violated triples: 8971/10000
	lr: 3.27099210816e-09
	Share Pull:  2773795328.0
	Share Push:  99859510.4865
Iter: 2
	Violated triples: 8971/10000
	lr: 3.30370202925e-09
	Share Pull:  2773520896.0
	Share Push:  99854714.3965
Iter: 3
	Violated triples: 8971/10000
	lr: 3.33673904954e-09
	Share Pull:  2773245440.0
	Share Push:  99849844.0609
Iter: 4
	Violated triples: 8971/10000
	lr: 3.37010644003e-09
	Share Pull:  2772967424.0
	Share Push:  99844914.7307
Iter: 5
	Violated triples: 8971/10000
	lr: 3.40380750443e-09
	Share Pull:  2772687872.0
	Share Push:  99839908.5081
Iter: 6
	Violated triples: 8971/10000
	lr: 3.43784557948e-09
	Share Pull:  2772407040.0
	Share Push:  99834854.5789
Iter: 7
	Violated triples: 8971/10000
	lr: 3.47222403527e-09
	Share Pull:  2772124416.0
	Share Push:  99829760.9938
Iter: 8
	Violated triples: 8971/10000
	lr: 3.50694627563e-09
	Share Pull:  2771838976.0
	Share Push:  99824585.8428
Iter: 9
	Violated triples: 8971/10000
	lr: 3.54201573838e-09
	Share Pull:  2771551232.0
	Share Push:  99819339.7996
Iter: 10
	Violated triples: 8971/10000
	lr: 3.57743589577e-09
	Share Pull:  2771261952.0
	Share Push:  99814051.013
Iter: 11
	Violated triples: 8971/10000
	lr: 3.61321025472e-09
	Share Pull:  2770970624.0
	Share Push:  99808757.9704
Iter: 12
	Violated triples: 8971/10000
	lr: 3.64934235727e-09
	Share Pull:  2770676224.0
	Share Push:  99803315.7512
Iter: 13
	Violated triples: 8971/10000
	lr: 3.68583578084e-09
	Share Pull:  2770379776.0
	Share Push:  99797832.9642
Iter: 14
	Violated triples: 8971/10000
	lr: 3.72269413865e-09
	Share Pull:  2770082304.0
	Share Push:  99792273.9956
Iter: 15
	Violated triples: 8971/10000
	lr: 3.75992108004e-09
	Share Pull:  2769781760.0
	Share Push:  99786689.8698
Iter: 16
	Violated triples: 8971/10000
	lr: 3.79752029084e-09
	Share Pull:  2769479168.0
	Share Push:  99780998.881
Iter: 17
	Violated triples: 8971/10000
	lr: 3.83549549375e-09
	Share Pull:  2769174016.0
	Share Push:  99775281.4463
Iter: 18
	Violated triples: 8971/10000
	lr: 3.87385044869e-09
	Share Pull:  2768866304.0
	Share Push:  99769516.4045
Iter: 19
	Violated triples: 8971/10000
	lr: 3.91258895317e-09
	Share Pull:  2768556800.0
	Share Push:  99763673.9912
Iter: 20
	Violated triples: 8971/10000
	lr: 3.9517148427e-09
	Share Pull:  2768244224.0
	Share Push:  99757740.9694
Iter: 21
	Violated triples: 8971/10000
	lr: 3.99123199113e-09
	Share Pull:  2767929088.0
	Share Push:  99751790.101
Iter: 22
	Violated triples: 8971/10000
	lr: 4.03114431104e-09
	Share Pull:  2767611136.0
	Share Push:  99745714.9366
Iter: 23
	Violated triples: 8971/10000
	lr: 4.07145575415e-09
	Share Pull:  2767290880.0
	Share Push:  99739587.7553
Iter: 24
	Violated triples: 8971/10000
	lr: 4.11217031169e-09
	Share Pull:  2766968832.0
	Share Push:  99733391.077
Iter: 25
	Violated triples: 8971/10000
	lr: 4.15329201481e-09
	Share Pull:  2766643200.0
	Share Push:  99727140.9666
Iter: 26
	Violated triples: 8971/10000
	lr: 4.19482493496e-09
	Share Pull:  2766316544.0
	Share Push:  99720809.7112
Iter: 27
	Violated triples: 8971/10000
	lr: 4.23677318431e-09
	Share Pull:  2765985536.0
	Share Push:  99714416.9488
Iter: 28
	Violated triples: 8971/10000
	lr: 4.27914091615e-09
	Share Pull:  2765652480.0
	Share Push:  99707988.4693
Iter: 29
	Violated triples: 8971/10000
	lr: 4.32193232531e-09
	Share Pull:  2765317120.0
	Share Push:  99701483.468
shape: (2720, 37)
train-acc: 9.890%                                
train confusion matrix:
 [[   0.    0.    0.    0.    0.    0.    0.  278.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  275.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  266.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  278.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  258.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  270.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  266.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  269.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  274.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  286.    0.    0.]]
test-acc: 10.234%
test confusion matrix:
 [[   0.    0.    0.    0.    0.    0.    0.  122.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  125.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  134.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  122.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  142.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  130.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  134.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  131.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  126.    0.    0.]
 [   0.    0.    0.    0.    0.    0.    0.  114.    0.    0.]]
neighbor cost: nan
Iter: 0
	Violated triples: 8973/10000
	lr: 4.36515164857e-09
	Share Pull:  26448406528.0
	Share Push:  1.36873093202e+11
Iter: 1
	Violated triples: 8973/10000
	lr: 4.40880316505e-09
	Share Pull:  26396981248.0
	Share Push:  1.36456324417e+11
Iter: 2
	Violated triples: 8973/10000
	lr: 4.4528911967e-09
	Share Pull:  26345385984.0
	Share Push:  1.36039692088e+11
Iter: 3
	Violated triples: 8973/10000
	lr: 4.49742010867e-09
	Share Pull:  26293499904.0
	Share Push:  1.35621517307e+11
Iter: 4
	Violated triples: 8973/10000
	lr: 4.54239430976e-09
	Share Pull:  26241298432.0
	Share Push:  1.35201406189e+11
Iter: 5
	Violated triples: 8973/10000
	lr: 4.58781825285e-09
	Share Pull:  26188806144.0
	Share Push:  1.3477996392e+11
Iter: 6
	Violated triples: 8973/10000
	lr: 4.63369643538e-09
	Share Pull:  26135963648.0
	Share Push:  1.34356430545e+11
Iter: 7
	Violated triples: 8973/10000
	lr: 4.68003339974e-09
	Share Pull:  26082787328.0
	Share Push:  1.33930894414e+11
Iter: 8
	Violated triples: 8973/10000
	lr: 4.72683373373e-09
	Share Pull:  26029277184.0
	Share Push:  1.33503765042e+11
Iter: 9
	Violated triples: 8973/10000
	lr: 4.77410207107e-09
	Share Pull:  25975457792.0
	Share Push:  1.33075222719e+11
Iter: 10
	Violated triples: 8973/10000
	lr: 4.82184309178e-09
	Share Pull:  25921234944.0
	Share Push:  1.32644784419e+11
Iter: 11
	Violated triples: 8973/10000
	lr: 4.8700615227e-09
	Share Pull:  25866637312.0
	Share Push:  1.32211818245e+11
Iter: 12
	Violated triples: 8973/10000
	lr: 4.91876213793e-09
	Share Pull:  25811664896.0
	Share Push:  1.31776874223e+11
Iter: 13
	Violated triples: 8973/10000
	lr: 4.96794975931e-09
	Share Pull:  25756340224.0
	Share Push:  1.31339695443e+11
Iter: 14
	Violated triples: 8973/10000
	lr: 5.0176292569e-09
	Share Pull:  25700618240.0
	Share Push:  1.30899792475e+11
Iter: 15
	Violated triples: 8973/10000
	lr: 5.06780554947e-09
	Share Pull:  25644498944.0
	Share Push:  1.3045733444e+11
Iter: 16
	Violated triples: 8973/10000
	lr: 5.11848360496e-09
	Share Pull:  25587984384.0
	Share Push:  1.3001279644e+11
Iter: 17
	Violated triples: 8973/10000
	lr: 5.16966844101e-09
	Share Pull:  25531105280.0
	Share Push:  1.29565942082e+11
Iter: 18
	Violated triples: 8973/10000
	lr: 5.22136512542e-09
	Share Pull:  25473814528.0
	Share Push:  1.29116298631e+11
Iter: 19
	Violated triples: 8973/10000
	lr: 5.27357877668e-09
	Share Pull:  25416093696.0
	Share Push:  1.28663828132e+11
Iter: 20
	Violated triples: 8973/10000
	lr: 5.32631456444e-09
	Share Pull:  25357991936.0
	Share Push:  1.28208951571e+11
Iter: 21
	Violated triples: 8973/10000
	lr: 5.37957771009e-09
	Share Pull:  25299460096.0
	Share Push:  1.27751160941e+11
Iter: 22
	Violated triples: 8973/10000
	lr: 5.43337348719e-09
	Share Pull:  25240514560.0
	Share Push:  1.27290589422e+11
Iter: 23
	Violated triples: 8973/10000
	lr: 5.48770722206e-09
	Share Pull:  25181132800.0
	Share Push:  1.26826995465e+11
Iter: 24
	Violated triples: 8973/10000
	lr: 5.54258429428e-09
	Share Pull:  25121320960.0
	Share Push:  1.26360443143e+11
Iter: 25
	Violated triples: 8973/10000
	lr: 5.59801013722e-09
	Share Pull:  25061070848.0
	Share Push:  1.25890924215e+11
Iter: 26
	Violated triples: 8973/10000
	lr: 5.6539902386e-09
	Share Pull:  25000378368.0
	Share Push:  1.25418420802e+11
Iter: 27
	Violated triples: 8973/10000
	lr: 5.71053014098e-09
	Share Pull:  24939272192.0
	Share Push:  1.24943375367e+11
