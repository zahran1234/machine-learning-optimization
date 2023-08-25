# Import Libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from random import random

import numpy as np
import math
import sys


(train_X, train_y), (test_X, test_y) = mnist.load_data()
y = train_y[:20000]
X = train_X[:20000]


def split(array, nrows, ncols):
    r, h = array.shape

    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def centroid(matrix, X, Y):
    y = 0
    x = 0
    counter = 0
    my_mat = []
    for i in range(X):
        for j in range(Y):
            if matrix[i][j] == 0:
                list = [[j, i]]
                my_mat.append([j, Y - i])
                counter += 1

    for i in my_mat:
        x = x + i[0]
        y = y + i[1]
    if counter == 0:
        return 0, 0

    return (x / counter), (y / counter)


new_data = np.empty((1, 32))

for phote in X:
    x2 = 0
    y2 = 0
    row = []
    list = split(phote, 7, 7)

    for i in list:
        x1, y1 = centroid(i, 7, 7)
        row.append(x1)
        row.append(y1)
    new_data = np.vstack([new_data, row])
new_data = np.delete(new_data, 0, 0)
print(" ")

[X_train, X_test, y_train, y_test] = train_test_split(new_data, y, test_size=0.1, random_state=44, shuffle=True)


class Neuron:
    def __init__(self, number_of_neuron):
        self.number_of_neuron = number_of_neuron
        self.net = 0
        self.output = 0
        self.delta = 1

    # to calculate net for neuron
    def set_net1(self, weight, value):
        self.net += weight * value
        self.output = 1 / (1 + math.e ** -self.net)

    # set output
    def set_output(self, value):
        self.output = value

    def set_delta(self, delta):
        self.delta = delta

    # to get number of this neuron in NN
    def get_number_of_neuron(self):
        return self.number_of_neuron

    # to get net of neuron
    def get_net(self):
        return self.net

    # to get output of neuron
    def get_output(self):
        return self.output


class Weight:
    def __init__(self, start, end, value):
        self.value = value
        self.start = start
        self.end = end

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_stare_end(self):
        return self.start, self.end


class Layer:

    def __init__(self, neurons, number_of_layer, start):
        self.neurons = neurons
        self.number_of_layer = number_of_layer
        self.list_for_neurons = []
        self.last_layer = False

        counter = start
        for i in range(neurons):
            n = Neuron(counter)
            self.list_for_neurons.append(n)
            counter += 1

    def get_number_of_neurons(self):
        return self.neurons

    def get_num_of_this_layer(self):
        return self.number_of_layer

    def get_list_of_neurons(self):
        return self.list_for_neurons


class NN:

    def __init__(self, layers, neurons,learning_rate):

        self.layers = layers
        self.neurons = neurons
        self.list_for_layers = []
        self.list_for_weight = []
        self.counter = 0
        self.number_of_labels = 0
        self.learning_rate=learning_rate

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        x, y2 = x_train.shape
        counter = 1
        layer = Layer(y2, 1, counter)  # remove the comment and replace frist 1 by y
        counter = y2  # counter =y

        #  layer number one
        self.list_for_layers.append(layer)
        counter += 1
        for i in range(self.layers):
            layer = Layer(self.neurons, i + 2, counter)
            self.list_for_layers.append(layer)
            counter += self.neurons
        # last layer
        self.number_of_labels = np.amax(y_train)

        layer = Layer(self.number_of_labels + 1, self.layers + 2, counter)
        layer.last_layer = True

        self.list_for_layers.append(layer)
        self.weights()

        # update weights

        for i in range(1):
            self.counter = 0

            while (1):

                self.forward(self.x_train[self.counter])  # self.counter
                self.back()

                self.counter += 1

                if self.counter >= 1:
                    return

                # if self.mean_square_error(self.list_for_layers[len(self.list_for_layers) - 1],  y_train[self.counter]) <= .2:
                # self.counter = 0

    def weights(self):
        count = .9
        counter = 1
        for i in self.list_for_layers:
            for x in i.list_for_neurons:

                for j in self.list_for_layers[counter].get_list_of_neurons():
                    w = Weight(x.get_number_of_neuron(), j.get_number_of_neuron(), random() / count)
                    count += count + 100

                    self.list_for_weight.append(w)
            counter += 1
            if counter == len(self.list_for_layers):
                break

    def forward(self, sample):
        for x in range(len(sample)):
            self.list_for_layers[0].list_for_neurons[x].output=sample[x]


        for i2 in range(len(self.list_for_layers) - 1):
            for x in self.list_for_layers[i2 + 1].list_for_neurons:
                for j in self.list_for_layers[i2].list_for_neurons:
                    current_weight_index = self.get_weight(j.number_of_neuron, x.number_of_neuron)
                    x.set_net1(self.list_for_weight[current_weight_index].value, j.output)
        softmax = self.softmax(self.list_for_layers[-1])
        count = 0
        for i in self.list_for_layers[-1].list_for_neurons:
            i.set_output(softmax[count])
            count += 1

        return softmax.index(max(softmax))

    def back(self):
        update = 0
        for i2 in range(len(self.list_for_layers) - 1, 0, -1):
            # at output layer
            if self.list_for_layers[i2].last_layer:

                for x in self.list_for_layers[i2].list_for_neurons:

                    out_J = x.output
                    current = self.list_for_layers[i2].list_for_neurons.index(x)

                    if current == self.y_train[self.counter]:
                        target_J = 1
                    else:
                        target_J = 0

                    delta_J = (target_J - out_J) * out_J * (1 - out_J)
                    end = x.number_of_neuron
                    x.delta = delta_J


                    for y in self.list_for_weight:

                        if y.end == end:
                            start = y.start
                            out_i = self.get_output_of_neuron(start)
                            delta_w = delta_J * out_i * self.learning_rate
                            self.list_for_weight[self.get_weight(start, end)].value += delta_w

                            """
                            print("update:-" , start , end )
                            print(target_J," target")
                            print(out_J, "out_J ")
                            print(delta_J,"delta",end)
                            print("--------------------")
                            """
                continue


            for i in self.list_for_layers[i2 - 1].list_for_neurons:
                for j in self.list_for_layers[i2].list_for_neurons:
                    sum2 = 0

                    for k in self.list_for_layers[i2 + 1].list_for_neurons:
                        index = self.get_weight(j.number_of_neuron, k.number_of_neuron)
                        sum2 += self.list_for_weight[index].value * k.delta

                    out_J = j.output
                    out_i = i.output
                    delta_j = sum2 *  (1 - out_J)*out_J

                    j.delta = delta_j
                    delta_w = delta_j * out_i * self.learning_rate
                    index2 = self.get_weight(i.number_of_neuron, j.number_of_neuron)
                    self.list_for_weight[index2].value += delta_w

    def mean_square_error_der(self, output_layer, real_output):
        counter = 0
        sum = 0
        for i in output_layer.list_for_neurons:
            if counter == real_output:
                sum += (i.output - real_output)
            else:
                sum += i.output
            counter += 1
        return (1 / self.number_of_labels) * sum

    def softmax(self, output_layer):

        list_for_softmax_values = []

        for iter in output_layer.list_for_neurons:
            if math.isnan(iter.output):
                list_for_softmax_values.append(0)
                continue

            list_for_softmax_values.append(iter.output)

        return list_for_softmax_values

    def score(self, X_test, Y_test):

        counter = 0
        counter_for_acc = 0
        for i in X_test:

            if self.forward(i) == Y_test[counter]:
                counter_for_acc += 1
            counter += 1

        return (counter_for_acc / counter) * 100

    def print_weight(self):

        for i in self.list_for_layers:
            for j in i.list_for_neurons:
                print(j.number_of_neuron)

    def get_weight(self, start, end):
        for i3 in range(len(self.list_for_weight)):
            if start == self.list_for_weight[i3].start:
                if end == self.list_for_weight[i3].end:
                    return i3


    def get_output_of_neuron(self, number_of_neuron):
        for i in self.list_for_layers:
            for x in i.list_for_neurons:
                if x.number_of_neuron == number_of_neuron:
                    return x.output


nn = NN(4, 4,.01)
nn.fit(X_train, y_train)