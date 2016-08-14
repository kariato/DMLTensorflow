# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import cPickle as pickle
import numpy as np
import tensorflow as tf

pickle_file = '/home/ubuntu/workspace/notMNIST.pickle'

image_size = 28
num_labels = 10
def reformat(dataset, labels):

  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# open pickle file
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print 'Training set', train_dataset.shape, train_labels.shape
  print 'Validation set', valid_dataset.shape, valid_labels.shape
  print 'Test set', test_dataset.shape, test_labels.shape

## Reformat into a shape that's more adapted to the models we're going to train:
## - data as a flat matrix,
## - labels as float 1-hot encodings.
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape



batch_size = 128
layer1_hidden_numnodes = 1024
num_classes = 10

graph = tf.Graph()
with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
#  weights = tf.Variable(
#    tf.truncated_normal([image_size * image_size, num_labels]))
#  biases = tf.Variable(tf.zeros([num_labels]))
    

  with tf.name_scope('hidden1'):
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, layer1_hidden_numnodes]))
    biases1 = tf.Variable(tf.zeros([layer1_hidden_numnodes]))
    #biases1 = tf.Variable(tf.zeros([128,1024]))
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    
    
  with tf.name_scope('softmax_linear'):
    weights2 = tf.Variable(tf.truncated_normal([layer1_hidden_numnodes, num_classes]))
    biases2 = tf.Variable(tf.zeros([num_classes]))
    logits2 = tf.matmul(hidden1, weights2) + biases2

#  tf.histogram_summary('weights', weights)
#  tf.histogram_summary('biases', biases)
  
  # Training computation.
#  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits2, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits2)
  
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)
  #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
  # Merge all the summaries and write them out to /tmp/mnist_logs
  merged = tf.merge_all_summaries()
  #writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)


num_steps = 6002
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print "Initialized"
  for step in xrange(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print "Minibatch loss at step", step, ":", l
      print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
      print "Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels)
  print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
