import sys
import numpy as np
import tensorflow as tf
source=sys.argv[1]
dest=sys.argv[2]
tp=int(sys.argv[3])
start=int(sys.argv[4])
end=int(sys.argv[5])
if tp==0:
    dtype=np.float64
else:
    dtype=np.float32

def save_tfrecords(source,dest,start,end,dtype):
    data=np.fromfile(source,dtype=dtype).reshape((-1,256,256))
    #res=[]
    with tf.compat.v1.python_io.TFRecordWriter(dest) as writer:
        #znum=z//128
        idx=0
        for k in range(start,end,128):
            zstart=k
            if end-zstart<128:
                break
            for i in range(2):
                for j in range(2):
                    xstart=i*128
                    ystart=j*128
                    datapoint=np.expand_dims(data[zstart:zstart+128,xstart:xstart+128,ystart:ystart+128],axis=-1)
     #               res.append(datapoint)
                    features = tf.train.Features(feature = { "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [datapoint.astype(dtype).tostring()])),"index":tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),'d0':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'d1':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'d2':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'c':tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))})     
                    example=tf.train.Example(features=features)
                    serialized=example.SerializeToString()
                    writer.write(serialized)
      #              np.array(res,dtype=dtype).tofile(npdest)
                    idx+=1

save_tfrecords(source,dest,start,end,dtype)

