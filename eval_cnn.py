import tensorflow as tf
import numpy as np
import gensim
import os
import math
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_folder", "./dataset_moviereviews/aclImdb/test/pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_folder", "./dataset_moviereviews/aclImdb/test/neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1535298694/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("dummy_data", False, "Use one Dummy sentence instead of the specified data")
tf.flags.DEFINE_boolean("do_backtracking", False, "Perform the backtracking for classification understanding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_class",2,"Number of classification classes")
tf.flags.DEFINE_boolean("input_embedded_data", True, "Specifies if the CNN should get already embedded data")
tf.flags.DEFINE_integer("max_input_size", 300, "The size of the preembedded input")
tf.flags.DEFINE_integer("embedding_size", 300, "The size of the preembedded input")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
# FLAGS(sys.argv)

# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
def calculate_backtrack_fcl(iim,v,w):
    new_iim = np.zeros([len(v),FLAGS.num_class])
    for i in range(len(new_iim)):
        for c in range(len(new_iim[0])):
            for j in range(len(iim)):
                new_iim[i,c] = sum([iim[j][c]*w[i,j]*v[i] for j in range(len(iim))])
    return new_iim

def calculate_backtrack_concat(iim,v,fnum,n):
    new_iim = np.reshape(iim,[n,fnum,FLAGS.num_class])
    new_v = np.reshape(v,[n,fnum])
    return (new_iim,new_v)

def calculate_backtrack_max_pool(iim,pooling_v,conv_v):
    result = []
    #print(str(iim[0]))
    for n in range(len(conv_v)):
        counter = 0
        print(str(conv_v[n].shape))
        new_iim = np.zeros([len(conv_v[n][0]),len(conv_v[n][0,0,0]),FLAGS.num_class])
        for i in range(len(conv_v[n][0])):
            for j in range(len(new_iim[0])):
                if pooling_v[n,j] == conv_v[n][0,i,0,j]:
                    #print("filterset:"+str(n)+" row:"+str(i)+", filter:"+str(j))
                    new_iim[i,j] = iim[n,j]
                    counter += 1
        #print(str(new_iim))
        print(str(counter))
        result.append(new_iim)
    return result

def calculate_backtrack_single_conv(iim,v,w):
    #print(str(iim[6]))
    new_iim = np.zeros([len(v),len(v[0]),FLAGS.num_class])
    len_w = len(w)
    print(iim.shape)
    print(v.shape)
    print(new_iim.shape)
    print(w.shape)
    for i in range(100): # words
        for j in range(len(new_iim[0])): # embeddings
            for c in range(len(new_iim[0,0])): # classifications
                for k in range(len(iim[0])): # filter
                    for m in range(min([i+1,len_w])):
                        new_iim[i,j,c] += iim[i+1-m,k,c]*v[i,j,0]*w[m,j,0,k]
    return new_iim

def calculate_backtrack_conv(iim,v,w):
    new_iim = []
    for n in range(len(iim)):
        new_iim.append(calculate_backtrack_single_conv(iim[n],v[0],w[n]))
    return new_iim

def sum_embedding(iim):
    new_iim = np.zeros([len(iim),len(iim[0,0])])
    for i in range(len(iim)):
        for j in range(len(iim[0])):
            for c in range(len(iim[0,0])):
                new_iim[i,c] += iim[i,j,c]
    return new_iim

def normalize_matrix(iim):
    # maximum = np.amax(iim)
    # minimum = np.amin(iim)
    # iim = (iim-minimum)/(maximum-minimum)
    # maximum_row_sum = max(sum([row[0] for row in iim]),sum([row[1] for row in iim]))
    # print(str(maximum_row_sum))
    # iim = iim/maximum_row_sum
    return iim

def create_output_string(dim,sentence,scores):
    dimsum_neg = sum([row[0] for row in dim])
    dimsum_pos = sum([row[1] for row in dim])
    result = "scores: pos: "+str(scores[0,1])+", neg: "+str(scores[0,0])+", DIM Sum: pos: "+ str(dimsum_pos) +", neg: "+str(dimsum_neg) + "Full review: "
    for i in range(len(dim)):
        if sentence[i] != "<UNK>":
            result += sentence[i]
            if dim[i,0] < dim[i,1]:
                result += "(pos:" + str(dim[i,1]) + ") "
            if dim[i,0] > dim[i,1]:
                result += "(neg:" + str(dim[i,0]) + ") "
    return result

def create_output_header(sentence, scores):
    return [sentence, "score pos:",str(scores[0,1]),"score neg: ",str(scores[0,0])]

# CHANGE THIS: Load data. Load your own data here
if FLAGS.dummy_data:
    x_raw = ["the movie is very good and I really enjoyed it"]
    y_test = [1]
else:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_folder,FLAGS.negative_data_folder)
    y_test = [y[1] for y in y_test]

if FLAGS.input_embedded_data:
    x_test = [[word for word in review.replace(".","").split(" ")] for review in x_raw]
else:
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

print("Loading Embedding model...")
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print("Embedding model loaded.")

def embedding_step(x_batch):
    allEmbedded = np.zeros([len(x_batch),FLAGS.max_input_size,300])
    for n, review in enumerate(x_batch):
        for i, word in enumerate(review):
            try:
                #embeddedWord = np.array(model[word])
                if i >= FLAGS.max_input_size:
                    break
                allEmbedded[n,i,:] = model[word]
            except KeyError:
                continue
    return allEmbedded

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        if FLAGS.do_backtracking:
            
            output = []
            # Load flags
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
            n = len(filter_sizes)
            fnum = FLAGS.num_filters
            # Generate batches for one epoch    
            batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)
            scores_tensor = graph.get_operation_by_name("output/scores").outputs[0]
            concat_v_tensor = graph.get_tensor_by_name("h_pool:0")
            output_w_tensor = graph.get_tensor_by_name("W:0")
            embedding_v_tensor = graph.get_tensor_by_name("embedding/emb:0")
            relu_v_tensors = [graph.get_tensor_by_name("conv-maxpool-"+str(f_size)+"/relu:0") for f_size in filter_sizes]
            conv_w_tensors = [graph.get_tensor_by_name("conv-maxpool-"+str(f_size)+"/W:0") for f_size in filter_sizes]
            for x_test_batch in batches:
                x_text = list(x_test_batch[0])
                if FLAGS.input_embedded_data:
                    x_test_batch = embedding_step(x_test_batch)
                scores,concat_v,output_w,emb_v = sess.run((scores_tensor,concat_v_tensor,output_w_tensor,embedding_v_tensor), {input_x: x_test_batch, dropout_keep_prob: 1.0})
                relu_v = [sess.run(relu_v_tensor ,{input_x: x_test_batch, dropout_keep_prob: 1.0}) for relu_v_tensor in relu_v_tensors]
                conv_w = [sess.run(conv_w_tensor ,{input_x: x_test_batch, dropout_keep_prob: 1.0}) for conv_w_tensor in conv_w_tensors]
                iim = [[scores[0,0],0],[0,scores[0,1]]]
                print(iim)
                iim = calculate_backtrack_fcl(iim,concat_v[0],output_w)
                iim,pooling_v = calculate_backtrack_concat(iim,concat_v[0],fnum,n)
                iim = calculate_backtrack_max_pool(iim,pooling_v,relu_v)
                iim = calculate_backtrack_conv(iim,emb_v,conv_w)
                iim = sum(iim)
                iim = sum_embedding(iim)
                dim = normalize_matrix(iim)
                print(str(dim))

                if FLAGS.input_embedded_data:
                    x_text.extend(["" for i in range(FLAGS.max_input_size - len(x_text[0]))])
                    sentence = x_text
                else:
                    blub = x_test_batch[0].reshape([-1,1])
                    sentence = list(vocab_processor.reverse(blub))

                neg = [str(word[0]) for word in dim]
                pos = [str(word[1]) for word in dim]

                output.append(create_output_header(x_raw[0],scores))
                output.append(["Words:"]+sentence)
                output.append(["pos:"]+pos)
                output.append(["neg:"]+neg)

            # Save the evaluation to a txt
            out_path = os.path.join(FLAGS.checkpoint_dir, "..", "backtracked_prediction3.csv")
            print("Saving evaluation to {0}".format(out_path))

            with open(out_path, 'w', newline='') as f:
                csv.writer(f,delimiter = ';').writerows(output)
        else:
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                if FLAGS.input_embedded_data:
                    x_test_batch = embedding_step(x_test_batch)
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            # Print accuracy if y_test is defined
            if y_test is not None:
                correct_predictions = float(sum(all_predictions == y_test))
                print("Total number of test examples: {}".format(len(y_test)))
                print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

            # Save the evaluation to a csv
            predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
            out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f,delimiter = ';').writerows(predictions_human_readable)