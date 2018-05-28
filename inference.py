import tensorflow as tf

class Siamese:
    contentMode = "vis"
    # Create model
    def __init__(self, contentMode, dtSize):
        self.contentMode = contentMode
        self.x1 = tf.placeholder(tf.float32, [None, dtSize])
        self.x2 = tf.placeholder(tf.float32, [None, dtSize])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 512, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 256, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")

        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

class SiameseDual(Siamese):
    contentMode = "cbvis"
    def __init__(self, contentMode, dtSizeCB, dtSizeVis):
        self.contentMode = contentMode
        self.x11 = tf.placeholder(tf.float32, [None, dtSizeCB])
        self.x12 = tf.placeholder(tf.float32, [None, dtSizeVis])
        
        self.x21 = tf.placeholder(tf.float32, [None, dtSizeCB])
        self.x22 = tf.placeholder(tf.float32, [None, dtSizeVis])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x11,self.x12)
            scope.reuse_variables()
            self.o2 = self.network(self.x21,self.x22)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x1, x2):
        weights = []
        fc11 = self.fc_layer(x1, 128, "fc1.1")
        ac11 = tf.nn.relu(fc11)
        
        fc12 = self.fc_layer(x2, 1024, "fc1.2")
        ac12 = tf.nn.relu(fc12)
        
        ac1 = tf.concat([ac11, ac12], 1)
        
        fc2 = self.fc_layer(ac1, 512, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 256, "fc3")
        return fc3