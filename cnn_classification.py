from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle

# Imports
import glob
import gensim
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk

#from tensorflow.contrib.data import Dataset, Iterator

tf.logging.set_verbosity(tf.logging.INFO)

def read_string_from_file(path):
    with open(file=path, mode='r',encoding="utf8") as myfile:
        data=myfile.read()
    return [word for word in data.replace(".","").split(" ")]

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.expand_dims(features["x"],-1)
    print(features["x"])
    batch_size = input_layer.shape[0].value
    word_number = input_layer.shape[1].value
    embedding_size = input_layer.shape[2].value

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        strides=[1,embedding_size],
        kernel_size=[3, embedding_size],
        padding="same",
        activation=tf.nn.relu)
    conv1 = tf.squeeze(conv1)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size= word_number,strides=word_number)
    pool1_flat = pool1

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        strides=[1,embedding_size],
        kernel_size=[2, embedding_size],
        padding="same",
        activation=tf.nn.relu)
    conv2 = tf.squeeze(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=word_number, strides=word_number)
    pool2_flat = pool2

    # Concat
    concat = tf.concat([pool1_flat, pool2_flat],1)
    # Dense Layer
    dense = tf.layers.dense(inputs=concat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    maxLength = 400
    # x = tf.placeholder(tf.float32,shape=[None,maxLength,300])
    # y = tf.placeholder(tf.int32)
    # # Load training and eval data
    # #posAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\aclImdb\\train\\pos\\*.txt")
    # posAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\smallTestSubset\\train\\pos\\*.txt")
    # posLabels = [1 for addr in posAddrs]
    # #negAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\aclImdb\\train\\neg\\*.txt")
    # negAddrs = glob.glob("C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\dataset_moviereviews\\smallTestSubset\\train\\neg\\*.txt")
    # negLabels = [0 for addr in negAddrs]

    # posStrings = [read_string_from_file(path) for path in posAddrs]
    # negStrings = [read_string_from_file(path) for path in negAddrs]
    # allData = posStrings+negStrings
    # labels = np.asarray(posLabels+negLabels, dtype=np.int32)
    # print("Files loaded")

    # #model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\David\\Documents\\TUHH\\SS18\\Research project\\word2vec-api-master\\GoogleNews-vectors-negative300.bin', binary=True)
    # model = np.random.rand(300)
    # print("Embedding model loaded")

    # allEmbedded = []
    # for review in allData:

    #     embedded = np.zeros([maxLength,300])
    #     for i, word in enumerate(review):
    #         try:
    #             #embeddedWord = np.array(model[word])
    #             if i >= maxLength:
    #                 break
    #             embedded[i,:] = model
    #         except KeyError:
    #             continue
    #     allEmbedded.append(np.array(embedded))

    # del model

    # print("All data embedded")
    # print("words: "+ str(len(allEmbedded[0])) +", embedded: " + str([len(x) for x in allEmbedded[0]]))
    #allEmbedded = np.array(allEmbedded)
    #train_data, eval_data, train_labels, eval_labels = sk.train_test_split(np.array(allEmbedded),labels,test_size=0.25, random_state = 42)
    filename = 'data/embedded_tfrecords'  # address to save the TFRecords file
   
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
        allEmbedded, labels = tf.train.shuffle_batch([embedded, label], batch_size=20, capacity=30, num_threads=1, min_after_dequeue=10)

    movieReviews_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/text_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": allEmbedded},
        y=labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True)
    movieReviews_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": allEmbedded},
        y=labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.app.run())

