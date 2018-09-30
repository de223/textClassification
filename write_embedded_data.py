# Imports
import glob
import gensim
import numpy as np
import tensorflow as tf

def read_string_from_file(path):
    with open(file=path, mode='r',encoding="utf8") as myfile:
        data=myfile.read()
    return [word for word in data.replace(".","").split(" ")]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def main(unused_argv):
    maxLength = 400
    # Load training and eval data
    #posAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\aclImdb\\train\\pos\\*.txt")
    posAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\smallTestSubset\\train\\pos\\*.txt")
    posLabels = [1 for addr in posAddrs]
    #negAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\aclImdb\\train\\neg\\*.txt")
    negAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\smallTestSubset\\train\\neg\\*.txt")
    negLabels = [0 for addr in negAddrs]

    posStrings = [read_string_from_file(path) for path in posAddrs]
    negStrings = [read_string_from_file(path) for path in negAddrs]
    allData = posStrings+negStrings
    labels = np.asarray(posLabels+negLabels, dtype=np.int64)
    print("Files loaded")

    #model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\word2vec-api-master\\GoogleNews-vectors-negative300.bin', binary=True)
    model = np.random.rand(300)
    print("Embedding model loaded")

    allNotEmbedded = []
    allEmbedded = []
    for review in allData:

        not_embedded = []
        embedded = np.zeros([maxLength,300])
        for i, word in enumerate(review):
            try:
                #embeddedWord = np.array(model[word])
                if i >= maxLength:
                    break
                embedded[i] = model
                not_embedded.append(word)
            except KeyError:
                continue
        allEmbedded.append(np.array(embedded))
        allNotEmbedded.append(not_embedded)

    del model
    filename = 'data/embedded_tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)

    for embedded,notembedded,label in zip(allEmbedded,allNotEmbedded,labels):
        # Create a feature
        embedded = embedded.flatten()
        embedded = embedded.astype(np.float)
        notembedded = np.asarray(notembedded)
        feature = {'train/label': _int64_feature(label),
                'train/embedded': _floats_feature(embedded),
                'train/notembedded': _bytes_feature(tf.compat.as_bytes(notembedded.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()

    with tf.Session() as sess:
        feature = {'train/label': tf.FixedLenFeature([], tf.int64),
            'train/embedded': tf.FixedLenFeature([], tf.string),
            'train/notembedded': tf.FixedLenFeature([], tf.string)}

        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        embedded = tf.decode_raw(features['train/embedded'], tf.float32)
        # Cast label data into int32
        label = tf.cast(features['train/label'], tf.int32)
        # Reshape image data into the original shape
        embedded = tf.reshape(embedded, [maxLength, 300])
        images, labels = tf.train.shuffle_batch([embedded, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
        print("done")


if __name__ == "__main__":
    tf.app.run()