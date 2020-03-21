import sys

from azure.iot.device import IoTHubDeviceClient, Message
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf


# load dataset
(
    (train_images, train_labels),
    (test_images, test_labels),
) = tf.keras.datasets.mnist.load_data()

# flatten images
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# add single timestep to training data
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]

# when testing our network with spiking neurons we will need to run it
# over time, so we repeat the input/target data for a number of
# timesteps.
n_steps = 30
test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))

with nengo.Network(seed=0) as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    nengo_dl.configure_settings(stateful=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node(np.zeros(28 * 28))

    # add the first convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
        inp, shape_in=(28, 28, 1)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # add the second convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
        x, shape_in=(26, 26, 32)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # add the third convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
        x, shape_in=(12, 12, 64)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # linear readout
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)

    # placeholder for iot communication
    azure_node = nengo.Node(lambda t, x: x, size_in=10, size_out=0)
    nengo.Connection(out, azure_node, synapse=0.1)

    out_p = nengo.Probe(out, label="out_p")

# some pre-trained weights are provided, or set `do_training=True` to train from scratch
do_training = False
if do_training:
    with nengo_dl.Simulator(net, minibatch_size=200) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
        )
        sim.fit(train_images, {out_p: train_labels}, epochs=10)

        # save the parameters to file
        sim.save_params("./mnist_params")


# set up communication with IoT hub
class IoTComm:
    def __init__(self, period):
        self.client = IoTHubDeviceClient.create_from_connection_string(sys.argv[1])
        self.period = period
        self.n_steps = 1

    def __call__(self, t, x):
        if self.n_steps % self.period == 0:
            message = Message("Detected digit: %d" % np.argmax(x))
            print(message)
            self.client.send_message(message)

        self.n_steps += 1


azure_node.output = IoTComm(n_steps)

with nengo_dl.Simulator(net) as sim:
    # load parameters
    sim.load_params("./mnist_params")

    for i in range(10):
        print("Ground truth:", test_labels[i, 0, 0])
        sim.predict(test_images[[i]])
