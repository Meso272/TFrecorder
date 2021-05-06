import sys
import os
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
d=np.zeros(((end-start)*32,128,128,128,1))

def save_tfrecords(source,dest,start,end,dtype):
   
    #res=[]
    a=0
    with tf.compat.v1.python_io.TFRecordWriter(dest) as writer:
        #znum=z//128
        idx=0
        for sid in range(start,end):
            filename="aramco-snapshot-%s.f32" % sid
            path=os.path.join(source,filename)
            data=np.fromfile(path,dtype=np.float32).reshape((449,449,235))
            for i in range(4):
                for j in range(4):
                    for k in range(2):
                        xstart=min(i*128,449-128)
                        ystart=min(j*128,449-128)
                        zstart=min(k*128,235-128)
                        datapoint=np.expand_dims(data[xstart:xstart+128,ystart:ystart+128,zstart:zstart+128],axis=-1)
                        d[idx]=datapoint
                        #print(idx)
                        a+=np.mean(datapoint)
     #               res.append(datapoint)
                        features = tf.train.Features(feature = { "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [datapoint.astype(dtype).tostring()])),"index":tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),'d0':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'d1':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'d2':tf.train.Feature(int64_list=tf.train.Int64List(value=[128])),'c':tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))})     
                        example=tf.train.Example(features=features)
                        serialized=example.SerializeToString()
                        writer.write(serialized)
      #              np.array(res,dtype=dtype).tofile(npdest)
                        idx+=1
        print(a/idx)

save_tfrecords(source,dest,start,end,dtype)
d=d.flatten()
print(d.size)
print(np.mean(d))
print(np.std(d))
