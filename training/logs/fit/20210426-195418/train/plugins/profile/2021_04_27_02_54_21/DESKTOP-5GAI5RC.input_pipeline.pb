  *�������@�����@2�
QIterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map	>�٬��@!��y�>nM@)������@1����M@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord	vOjM@!l��Q`B@)vOjM@1l��Q`B@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4	���B�i�?!��,Ax�?)���B�i�?1��,Ax�?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap	j�q���@![	�A�B@)j�q���?1�U'��n�?:Preprocessing2�
[Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch	*��Dذ?!��?sܬ�?)*��Dذ?1��?sܬ�?:Preprocessing2�
{Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl::ParallelMapV2	ڬ�\mŮ?!�$�'��?)ڬ�\mŮ?1�$�'��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl	���S㥻?!�(ޛ�?)���JY��?1YkR(���?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality	lxz�,C�?!���Y�F�?)�MbX9�?1�7�\9�?:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch=�U����?!��o1��?)=�U����?1��o1��?:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch�0�*��?!�/��޴�?)�0�*��?1�/��޴�?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::ForeverRepeat::Prefetch::MapAndBatch::Map::Prefetch::MemoryCache	���?!k!�����?)S�!�uq�?1mʪs�x�?:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::ForeverRepeatݵ�|г�?!����L�?){�G�zt?1�1����?:Preprocessing2F
Iterator::Model���B�i�?!��,Ax�?)��_�Le?1x��⫝̸?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism��6��?!��U���?)����Mb`?1\�>��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.