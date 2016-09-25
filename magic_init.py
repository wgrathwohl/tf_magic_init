import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import re

def initialize_weight_pca(x, out_features, whiten=False):
    # x is [batch_size, n_input_features]
    # if not enough out_features for pca then just return N(0, 1) weights
    if x.shape[0] < out_features or x.shape[1] < out_features:
        print("Not enough data for PCA estimation, using Gaussian Initialization")
        return np.random.normal(0, 1, (x.shape[1], out_features))
    pca = PCA(n_components=out_features, whiten=whiten)
    pca.fit(x)
    weights = pca.components_.T
    return weights

def initialize_weight_zca(x, out_features):
    return initialize_weight_pca(x, out_features, True)

def initialize_weight_kmneans(x, out_features):
    # if not enough out_features for pca then just return N(0, 1) weights
    if x.shape[0] < out_features or x.shape[1] < out_features or x.shape[0] < x.shape[1]:
        print("Not enough data for PCA estimation, using Gaussian Initialization")
        return np.random.normal(0, 1, (x.shape[1], out_features))
    # whiten data
    pca = PCA(n_components=x.shape[1], whiten=True)
    whitened_x = pca.fit_transform(x)
    km = MiniBatchKMeans(n_clusters=out_features, batch_size=10*out_features).fit(whitened_x)
    weights = km.cluster_centers_.T

    return weights

def unwrap_conv_tensor(x, filter_shape):
    # unwraps a 3d tensor to 2d so we can apply kmeans or pca to initialize our weights
    # each example is [h, w, c]
    fh, fw = filter_shape
    h, w, c = x[0].shape
    n_rows = h - fh + 1
    n_cols = w - fw + 1
    unwrapped = []
    for example in x:
        for i in range(n_rows):
            for j in range(n_cols):
                patch = example[i:i+fh, j:j+fw, :]
                patch_unwrapped = np.reshape(patch, (1, -1))
                unwrapped.append(patch_unwrapped)
    return np.concatenate(unwrapped)

def initialize_network_weights(layers, init_type, sess, batch_func, batch_pl):
    """
    initializes the weights of our network.
    layers: list of layer parameters in order of input -> output
    init_type: type of init to use (kmeans recommended)
    sess: tensorflow session with model already defined
    batch_fun: function that takes no arguments, should return a new batch of input data
    batch_pl: list of tensorflow placeholders, should align with results of batch_func()
    """
    assert init_type in ("pca", "zca", "kmeans"), "unsupported init type"
    for (W, b, pre_act, post_act, s, layer_input) in layers:
        # get layer input values
        x, = sess.run([layer_input], feed_dict={bpl: b for bpl, b in zip(batch_pl, batch_func())})
        # if conv layer, flatten tensor to matrix
        conv = False
        if len(x.shape) == 4:
            fh, fw, in_features, out_features = W.get_shape().as_list()
            conv = True
            x = unwrap_conv_tensor(x, (fh, fw))
        else:
            in_features, out_features = W.get_shape().as_list()

        if init_type == "pca":
            weight_value = initialize_weight_pca(x, out_features)
        elif init_type == "zca":
            weight_value = initialize_weight_zca(x, out_features)
        elif init_type == "kmeans":
            weight_value = initialize_weight_kmneans(x, out_features)
        else:
            assert False
        if conv:
            weight_value = weight_value.reshape((W.get_shape().as_list()))

        W_assign_op = W.assign(weight_value)
        sess.run(W_assign_op)

def layerwise_within_layer_init(pre_act_mean, pre_act_var, W, b, sess, batch_func, batch_pl, beta=0.0):
    # z mean and z var must have shape [1, W.shape[1]] for linear layers
    # [1, 1, 1, W.shape[3] for conv layers, i.e. keep_dims=True
    W_assign_op = W.assign(W * tf.rsqrt(pre_act_var))
    b_assign_op = b.assign((tf.ones_like(b) * beta) - tf.squeeze(pre_act_mean * tf.rsqrt(pre_act_var)))
    with tf.control_dependencies([W_assign_op, b_assign_op]):
        op = tf.no_op()
    sess.run(op, feed_dict={bpl: b for bpl, b in zip(batch_pl, batch_func())})

def C_tilde(fake_loss, W, b):
    # since tf adds over the first dim for gradient calculations, need to split into a tensor
    # for each loss
    [W_grads] = tf.gradients(fake_loss, W)
    C = tf.sqrt(tf.reduce_mean(tf.square(W_grads)) / tf.reduce_mean(tf.square(W)))

    return C

def within_layer_init(layers, sess, batch_func, batch_pl, beta=0.0):
    """
    Scales weights and sets biases so all pre_act values have mean 0 and variance 1

    All layers should be in the form z = f(s*(xW + b)), where pre_act = s(xW + b)
    and post_act = f(s*(xW + b))
    where f is some element-wise nonlinearity and s is a scale factor initialized to 1
    assumes all biases initialized to 0
    assumes all scale factors initialized to 1
    each value in layers is a tuple consiting of (W, b, pre_act, post_act, s, layer_input)
    """

    for (W, b, pre_act, post_act, s, layer_input) in layers:
        # if linear layer
        if len(pre_act.get_shape().as_list()) == 2:
            # get pre_act moments
            mu, var = tf.nn.moments(pre_act, [0], keep_dims=True)
        elif len(pre_act.get_shape().as_list()) == 4:
            # get pre_act moments
            mu, var = tf.nn.moments(pre_act, [0, 1, 2], keep_dims=True)
        else:
            assert False

        # run init for layer
        layerwise_within_layer_init(mu, var, W, b, sess, batch_func, batch_pl, beta=beta)

def between_layer_init(layers, sess, batch_func, batch_pl, iters=10, alpha=.25, fake_loss=None):
    """
    Scales the weights and biases of all layers such that the expected learning rate is
    roughly the same across all layers

    layers: list of layer params
    sess: tensorflow session with model initialized
    batch_func: function with no args that returns a new data batch
    batch_pl: list of tensorflow placeholders that your model needs to run
    iters: number of iterations to use
    alpha: smoothing factor on iterative optimization
    fake_loss: if your model has a standard structure, then leave this as None and we will create
        a fake loss. If your model does not follow this structure, for example, if it has multiple
        outputs or loss functions, then you should create your own random gaussian loss and pass
        it in here.
    """

    # get fake regression target with gives gradients distributed as N(0, 1)
    if fake_loss is None:
        last_pre_act = layers[-1][2]
        fake_target = tf.random_normal(tf.shape(last_pre_act))
        fake_loss = tf.reduce_mean(tf.reduce_sum(fake_target * last_pre_act, [1]))

    # setup placeholders for rk values and value change assignments
    rk_pls = [tf.placeholder("float") for _ in layers]
    W_assign_ops, b_assign_ops, s_assign_ops = [], [], []
    for (W, b, pre_act, post_act, s, layer_input), rk_pl in zip(layers, rk_pls):
        W_assign_ops.append(W.assign(rk_pl * W))
        b_assign_ops.append(b.assign(rk_pl * b))
        s_assign_ops.append(s.assign(tf.div(s, rk_pl)))

    # get all C_tiles (this is the expected learning rate per layer)
    C_tildes = [C_tilde(fake_loss, W, b) for W, b, pre_act, post_act, s, layer_input in layers]

    for i in range(iters):
        # evaluate C_tildes
        ct_values = sess.run(C_tildes, feed_dict={bpl: b for bpl, b in zip(batch_pl, batch_func())})
        print("Ct values: {}".format(ct_values))
        # get geometric mean of values
        logs = np.log(ct_values)
        exp = logs.mean()
        C = np.e**exp
        # get rk for each layer
        rks = [(Ck/C)**(alpha/2.0) for Ck in ct_values]

        # setup feed dict for updating values
        fd = {rk_pl: rk for rk_pl, rk in zip(rk_pls, rks)}
        sess.run(W_assign_ops + b_assign_ops + s_assign_ops, feed_dict=fd)

def magic_init(layers, init_type, sess, batch_func, init_batch_pl, fake_loss=None):
    """
    Runs the whole pipline of
        1) Randomly initialize weights via either gaussian, pca, or kmeans
        2) Scale weights and set biases so that pre-activations have mean 0 and variance 1
        3) Iteratively scale the weights and biases of each layer so that all layers learn at
            roughly the same rate

    Your model must consist of layers in the form of z = f(s(x*W + b)) where f is some
        parameterless nonlinearity (relu, lrelu, tanh, softmax, sigmoid, ...). We need a scale parameter s which is a [1]-shape float tensor. This is so that we can undo the scaling
        that is applied in step 3

    layers: list of layer params. Each element is a tuple of the form:
        W: (weight matrix or kernel),
        b: (bias vector),
        pre_act: result of s(xW + b),
        post_act: layer output,
        s: Scale factor,
        layer_input: The tensor that the layer operates on
    init_type: either "pca", "zca", or "kmeans", determines the type or random initialization used
    sess: You tensorflow session with your model already initialized
    batch_func: Function with no args that returns a new data batch. For best results, should
        return a new batch every time it is called
    init_batch_pl: A list of tensorflow placeholders. Should contain all data needed to run a
        prediction step
    fake_loss: If your model has a standard structure, then leave this as None and we will create
        a fake loss. If your model does not follow this structure, for example, if it has multiple
        outputs or loss functions, then you should create your own random gaussian loss and pass
        it in here.
    """
    initialize_network_weights(layers, init_type, sess, batch_func, init_batch_pl)
    within_layer_init(layers, sess, batch_func, init_batch_pl)
    between_layer_init(layers, sess, batch_func, init_batch_pl, fake_loss=fake_loss)

##################################################################################################
# Auxillury helper code, not needed to use the magic_init
##################################################################################################
def load_cifar_data(cifar_data_path='/Users/grathwohl1/Code/ulfv_main/grathwohl/data/cifar-10-batches-py'):
    # helper to load in cifar data
    import cPickle
    d1 = '{}/data_batch_1'.format(cifar_data_path)
    d2 = '{}/data_batch_2'.format(cifar_data_path)
    d3 = '{}/data_batch_3'.format(cifar_data_path)
    d4 = '{}/data_batch_4'.format(cifar_data_path)
    d5 = '{}/data_batch_5'.format(cifar_data_path)
    d6 = '{}/test_batch'.format(cifar_data_path)
    print '... loading data'

    # Load the dataset
    f1 = open(d1, 'rb')
    train_set_1 = cPickle.load(f1)
    f1.close()
    f2 = open(d2, 'rb')
    train_set_2 = cPickle.load(f2)
    f2.close()
    f3 = open(d3, 'rb')
    train_set_3 = cPickle.load(f3)
    f3.close()
    f4 = open(d4, 'rb')
    train_set_4 = cPickle.load(f4)
    f4.close()
    f5 = open(d5, 'rb')
    train_set_5 = cPickle.load(f5)
    f5.close()

    f_train = open(d6, 'rb')
    test_set = cPickle.load(f_train)
    f_train.close()

    train_set_x = np.vstack((train_set_1['data'],train_set_2['data']))
    train_set_x = np.vstack((train_set_x,train_set_3['data']))
    train_set_x = np.vstack((train_set_x,train_set_4['data']))
    train_set_x = np.vstack((train_set_x,train_set_5['data']))
    train_set_x = train_set_x.reshape((-1,3,32,32)).transpose([0,2,3,1])
    train_set_x = np.asarray((train_set_x)/255., dtype='float32')

    train_set_y = train_set_1['labels']
    train_set_y = train_set_y + train_set_2['labels']
    train_set_y = train_set_y + train_set_3['labels']
    train_set_y = train_set_y + train_set_4['labels']
    train_set_y = train_set_y + train_set_5['labels']
    train_set_y = np.asarray(train_set_y)

    print "min", np.max(train_set_x)
    print "max", np.min(train_set_x)
    idx_list = np.arange(len(train_set_y))
    np.random.shuffle(idx_list)

    valid_set_x = train_set_x[idx_list[0:2000],:,:,:]
    valid_set_y = train_set_y[idx_list[0:2000]]
    train_set_x = train_set_x[idx_list[2000:],:,:,:]
    train_set_y = train_set_y[idx_list[2000:]]

    test_set_x = test_set['data'].reshape((-1,3,32,32)).transpose([0,2,3,1])

    test_set_x = (test_set_x)/255.
    test_set_y = np.asarray(test_set['labels'])

    print len(train_set_y), len(valid_set_y), len(test_set_y)


    width = 32
    height = 32

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), width, height, 3, 10]

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    with tf.device("/cpu:0"):
        tf.histogram_summary(x.op.name + '/activations', x)
        tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _create_variable(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _variable_with_weight_decay(name, shape, wd, stddev="MSFT", mean=0.0):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    if stddev == "MSFT":
        # use microsoft initialization
        stddev = microsoft_initilization_std(shape)
    var = _create_variable(
        name, shape,
        tf.truncated_normal_initializer(mean=mean, stddev=stddev)
    )
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1), nonlinearity=tf.nn.relu, return_params=False):
    """
    A Standard convolutional layer
    Assumes that state_below is 4d tensor with shape (batch_size, height, width, channels)
    Computes the function nonlinearity(s(x*W + b))
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape().as_list()[3]
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            wd=wd, stddev=stddev
        )
        scale = _create_variable("scale", [1], tf.constant_initializer(1.0), trainable=False)
        conv = tf.nn.conv2d(state_below, kernel, [1, filter_stride[0], filter_stride[1], 1], padding="SAME")
        biases = _create_variable("biases", [n_outputs], tf.constant_initializer(0.0))
        pre_act = scale * (conv + biases)
        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
    if not return_params:
        return output
    else:
        # Gotta make identity tensor so it can be fetched from the session
        if state_below.op.node_def.op == u'Placeholder':
            inp = tf.identity(state_below)
        else:
            inp = state_below
        return output, (kernel, biases, pre_act, output, scale, inp)

def linear_layer(state_below, scope_name, n_outputs, stddev, wd, nonlinearity=tf.nn.relu, return_params=False):
    """
    Standard linear neural network layer
    """
    if nonlinearity is None:
        nonlinearity = tf.identity

    n_inputs = state_below.get_shape().as_list()[1]
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay(
            'weights', [n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        scale = _create_variable("scale", [1], tf.constant_initializer(1.0), trainable=False)
        biases = _create_variable(
            'biases', [n_outputs], tf.constant_initializer(0.0)
        )
        pre_act = scale * tf.nn.xw_plus_b(state_below, weights, biases)

        output = nonlinearity(pre_act, name=scope.name)
        if not tf.get_variable_scope().reuse:
            _activation_summary(output)
    if not return_params:
        return output
    else:
        if state_below.op.node_def.op == u'Placeholder':
            inp = tf.identity(state_below)
        else:
            inp = state_below
        return output, (weights, biases, pre_act, output, scale, inp)

if __name__ == "__main__":
    # test code for this schmat, get cifar data at https://www.cs.toronto.edu/~kriz/cifar.html
    batch_size = 64
    # set up placeholders
    x_pl = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_pl = tf.placeholder(tf.int64, [None])

    # A stupid, but deep model that is intentionally difficult to optimize
    # via vanilla gradient descent
    l1, l1_params = conv_layer(
        x_pl, "conv1", 64, (5, 5), .01, .004, filter_stride=(2, 2), return_params=True
    )
    l2, l2_params = conv_layer(
        l1, "conv2", 128, (5, 5), .01, .004, filter_stride=(2, 2), return_params=True
    )
    l3, l3_params = conv_layer(
        l2, "conv3", 256, (3, 3), .01, .004, filter_stride=(2, 2), return_params=True
    )
    l4, l4_params = conv_layer(
        l3, "conv4", 256, (3, 3), .01, .004, filter_stride=(1, 1), return_params=True
    )
    l5, l5_params = conv_layer(
        l4, "conv5", 256, (3, 3), .01, .004, filter_stride=(1, 1), return_params=True
    )
    l5_flat = tf.reshape(l5, [-1, np.prod(l5.get_shape().as_list()[1:])])
    fc4, fc4_params = linear_layer(l5_flat, "fc4", 1024, .01, .004, return_params=True)
    fc5, fc5_params = linear_layer(fc4, "fc5", 512, .01, .004, return_params=True)
    logits, logits_params = linear_layer(
        fc5, "logits", 10, .01, .004, return_params=True, nonlinearity=None
    )
    # create loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_pl))
    optimizer = tf.train.GradientDescentOptimizer(1e-3)

    grads = optimizer.compute_gradients(loss, tf.trainable_variables())
    train_step = optimizer.apply_gradients(grads)
    for (g, v) in grads:
        tf.histogram_summary("{}_gradients".format(v.name), g)
    summary_op = tf.merge_all_summaries()


    correct_prediction = tf.equal(tf.argmax(logits, 1), y_pl) # vector of bools
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter("/tmp/test_magic_init", graph_def=sess.graph_def)

    # get the MNIST data set
    cifar_x, cifar_y = load_cifar_data()[0]

    layers = [l1_params, l2_params, l3_params, l4_params, l5_params, fc4_params, fc5_params, logits_params]

    init_batch = cifar_x[:batch_size]
    batch_func = lambda: [init_batch]

    # Let that magic happen (try commenting this out to see the difference)
    magic_init(layers, "kmeans", sess, batch_func, [x_pl])

    num_epochs = 1000
    for i in range(num_epochs):
        batch_xs = cifar_x[batch_size * i: batch_size * (i + 1)]
        batch_ys = cifar_y[batch_size * i: batch_size * (i + 1)]
        _, loss_v = sess.run([train_step, loss], feed_dict={x_pl: batch_xs, y_pl: batch_ys})
        if loss_v < .1:
            print(i)
            break
        if i % 10 == 0:
            ac, sum_str = sess.run([accuracy, summary_op], feed_dict={x_pl: batch_xs, y_pl: batch_ys})
            print("{} | Accuracy: {}, Loss: {}".format(i, ac, loss_v))
            summary_writer.add_summary(sum_str, i)










