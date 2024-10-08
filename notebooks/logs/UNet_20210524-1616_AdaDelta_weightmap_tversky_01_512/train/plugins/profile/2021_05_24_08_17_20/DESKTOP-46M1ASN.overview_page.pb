�	f0F$��@f0F$��@!f0F$��@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCf0F$��@_Cp\�A2@1�N�V"�@A�o}XoԲ?I�%�L1g@rEagerKernelExecute 0*	     �J@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch���_vO�?!4>2�ީK@)���_vO�?14>2�ީK@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�~j�t��?!�w�ZnV@)HP�sג?1��4>2A@:Preprocessing2F
Iterator::Model1�Zd�?!      Y@)�I+�v?1�B�(��$@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI��p�]W@Qly�E%X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	_Cp\�A2@_Cp\�A2@!_Cp\�A2@      ��!       "	�N�V"�@�N�V"�@!�N�V"�@*      ��!       2	�o}XoԲ?�o}XoԲ?!�o}XoԲ?:	�%�L1g@�%�L1g@!�%�L1g@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��p�]W@yly�E%X@�"g
9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb�r�?!b�r�?08"s
Dgradient_tape/model/up_sampling2d_3/resize/ResizeNearestNeighborGradResizeNearestNeighborGrad�%��a+�?!b:q\4D�?"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad�"�M���?!�o۩�?"g
9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�0��D�?!U�!���?08"h
>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3FusedBatchNormGradV3\�?!��Z�Y�?"e
8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput���?!�����?08"g
9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�G�6��?!Mm��?08"6
model/conv2d_16/Conv2DConv2D���/�ǔ?!��zw^�?08"e
8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputConv2DBackpropInputn�P��?!=�u�L��?08"e
8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputConv2DBackpropInput�8�s
J�?!Ǫ�-���?08I��u�C@Qu���wN@YsusJ�, @aTd�E�~X@q���|��U@y�;����3?"�	
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�87.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 