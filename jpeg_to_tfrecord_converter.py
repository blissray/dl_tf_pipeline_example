import os
import numpy as np
import tensorflow as tf

import pprint
pp = pprint.PrettyPrinter()


TRAIN  = "train"
TEST  = "test"

flags = tf.app.flags
flags.DEFINE_string("image_dir", "Images", "The directory of dog images [Images]")
flags.DEFINE_string("output_dir", "tfrecords", "The directory of tfrecord_output [tfrecords]")
flags.DEFINE_boolean("cropping", "False", "The boolean vairable of dog faces cropping [False]")
flags.DEFINE_integer("image_height", "256", "The boolean vairable of dog faces cropping [350]")
flags.DEFINE_integer("image_width", "256", "The boolean vairable of dog faces cropping [350]")
flags.DEFINE_boolean("image_adjusted", "False", "The boolean vairable expressing whether or not to reduce the image without distorting the image according to the face size of the dog [False]")
flags.DEFINE_boolean("image_augumentation", "False", "The boolean vairable of generating image data added with random distortion, upside-downside, side-to-side reversal, etc. [False]")
flags.DEFINE_float("test_ratio", "0.2", "The ratio of test image data set [0.8]")

FLAGS = flags.FLAGS

def get_total_data():
    """ image가 저장된 폴더로 부터 모든 JPEG를 가져와서 [파일명, 개품중(Bleed)]의 형태의 Numpy Array를 생성함
    image폴더의 기본 저장형태는 "Images\개품종명\파일명" 형태로 저장되어 있다고 가정함

    Returns:
        Numpy.ndarray: [[파일명, name_of_dog_bleed]]
        [['n02085620_10074.jpg', 'n02085620-Chihuahua'],
        ['n02085620_10131.jpg', 'n02085620-Chihuahua'],
        ['n02085620_10621.jpg', 'n02085620-Chihuahua']

    """
    IMAGE_DIR = FLAGS.image_dir

    bleeds = os.listdir(IMAGE_DIR)
    image_set = {}

    for bleed in bleeds:
        image_dir = os.path.join(IMAGE_DIR, bleed)
        image_set[bleed] = os.listdir(image_dir)

        total_data = []
        for bleed in image_set.keys():
            total_data.extend([ [filename,bleed] for filename in image_set[bleed]])
        total_data = np.array(total_data)
    return total_data

def get_splitted_data(total_data):
    """
    Returns:
    """

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
         total_data[:,0], total_data[:,1], test_size=FLAGS.test_ratio, random_state=1)
    return X_train, X_test, y_train, y_test

def generate_patches():
    with open('testfile.txt', 'r') as f:
        for patch in f.readlines():
            yield patch[:-1]

def persistence_image_data_to_tfrecords(x_data, y_data, data_type, split_index=256):
    """
    Returns:
    """

    OUTPUT_DIR = os.path.join(FLAGS.output_dir,data_type)
    IMAGE_DIR = FLAGS.image_dir

    if not(os.path.exists(FLAGS.output_dir)):
        os.mkdir(FLAGS.output_dir)

    if not(os.path.exists(OUTPUT_DIR)):
        os.mkdir(OUTPUT_DIR)
        print("Directory create : {0}".format(OUTPUT_DIR,))

    writer = None
    sess = None
    current_index = 0



    for images_filename,breed in zip(x_data,y_data):
        if not(images_filename[-3:] == "jpg"):
            print("Error - ", images_filename)
            continue
        if current_index % split_index == 0:
            if writer:
                writer.close()
            if sess:
                sess.close()
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            sess = tf.Session(graph=graph)
            sess.run(tf.global_variables_initializer())

            record_filename = "{output_dir}/{data_type}-{current_index}.tfrecords".format(
                output_dir=OUTPUT_DIR, data_type=data_type, current_index=current_index
            )
            writer = tf.python_io.TFRecordWriter(record_filename)

        file_full_path = os.path.join(IMAGE_DIR, breed,  images_filename)
        image_file = tf.read_file(file_full_path)
        try:
            print(file_full_path)
            image = tf.image.decode_jpeg(image_file)
        except InvalidArgumentError as e:
            print(e)
            print("Error : ", images_filename)
            continue


        grayscale_image = tf.image.rgb_to_grayscale(image)
        resized_image = tf.image.resize_images(grayscale_image, [FLAGS.image_width, FLAGS.image_height])

        image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
        image_label = breed.encode("utf-8")

        example = tf.train.Example(features = tf.train.Features(
                                    feature={'label':
                                              tf.train.Feature(bytes_list=tf.train.BytesList(
                                                  value=[image_label])),
                                              "images":
                                              tf.train.Feature(bytes_list=tf.train.BytesList(
                                                  value=[image_bytes]))
                                             }
                                  ))
        writer.write(example.SerializeToString())
        current_index += 1
    writer.close()

def main(_):
    print ('Converting JPEG to tfrecord datatype')
    print ('Argument setup')
    pp.pprint(flags.FLAGS.__flags)
    print ('---------------------------------')

    total_data = get_total_data()
    number_of_data_types = len(np.unique(total_data[:, 1]))
    print("The number of data : {0}".format(total_data.shape[0],))
    print("The number of bleeds : {0}".format(number_of_data_types,))

    print('---------------------------------')
    X_train, X_test, y_train, y_test = get_splitted_data(total_data)
    print("Train / Test ratio : {0:.2f} / {1:.2f}".format( 1-FLAGS.test_ratio, FLAGS.test_ratio ))
    print("Number of train data set : {0}".format(len(X_train)))
    print("Number of test data set : {0}".format(len(X_test)))


    print('---------------------------------')
    persistence_image_data_to_tfrecords(X_train, y_train, data_type=TRAIN)
    persistence_image_data_to_tfrecords(X_test, y_test, data_type=TEST)

    #TODO - Google Detection API 써서 실험 먼저 해보기
    #TODO - Test에도 공동적용해야할 내용 ==> data resize + adjustable
    #TODO - 데이터 리사이즈 ==> adjustable에 맞춰 처리함
    #TODO - cropping 여부 확인 ==> traing 데이터만 처리,
    #TODO - 일단 테스트 데이터 먼저 리사이즈 처리?
    #TODO - Training 데이터의 data augumentation을 어떻게 할 것인가?


if __name__ =="__main__":
    tf.app.run()
