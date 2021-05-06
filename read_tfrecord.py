import tensorflow as tf

def _parse_64(example_proto):
    features={"data":tf.FixedLenFeature((),tf.string)}
    parsed_features=tf.parse_single_example(example_proto,features)
    data=tf.decode_raw(parsed_features['data'],tf.float64)
    return data

def _parse_32(example_proto):
    features={"data":tf.FixedLenFeature((),tf.string)}
    data=tf.decode_raw(parsed_features['data'],tf.float32)
    return data

def load_tfrecords_64(srcfile):
#srcfile: input tfrecord file
    dataset=tf.data.TFRecordDataset(srcfile)
    dataset=dataset.map(_parse_64)
    return dataset
def load_tfrecords_32(srcfile):
    dataset=tf.data.TFRecordDataset(srcfile)
    dataset=dataset.map(_parse_32)
    return dataset
