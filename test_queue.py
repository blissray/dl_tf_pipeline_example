import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_my_file_format(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'images': tf.FixedLenFeature([], tf.string),
        })

      # Convert from a string to a vector of uint8 that is record_bytes long.
    record_image = tf.decode_raw(features['images'], tf.uint8)
    image = tf.reshape(record_image, [256, 256, 1])
    label = tf.cast(features['label'], tf.string)
    return image, label


filenames=[]

import os
for file_name in os.listdir("./tfrecords/train/"):
    filenames.append(os.path.join("./tfrecords/train/",file_name))
batch_size=128
num_epochs=100


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    filename_queue = tf.train.string_input_producer(
            filenames,  shuffle=True)

    image, label = read_my_file_format(filename_queue)

    # min_after_dequeue
    # - 무작위로 샘플링 할 버퍼의 크기를 정의
    # - 크면 shuffling이 더 좋지만 느리게 시작되고 메모리가 많이 사용됨
    # capacity
    # min_after_dequeue보다 커야하며 더 큰 금액은 프리 페치 할 최대 값을 결정합니다.
    # 추천: min_after_dequeue + (num_threads + 약간의 여유값) * batch_size

    min_after_dequeue = 1024
    batch_size = batch_size
    capacity = min_after_dequeue + 4 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while True:
            example = sess.run([image_batch])
            print(example)
    except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
