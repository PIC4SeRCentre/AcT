# Data loading and TF Dataset creation

X_train, y_train, X_test, y_test = load_mpose(
    self.config["DATASET"], self.split, verbose=False
)
self.train_len = len(y_train)
self.test_len = len(y_test)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=self.config["VAL_SIZE"],
    random_state=self.config["SEEDS"][self.fold],
    stratify=y_train,
)  # <- IMPORTANTE!

self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
self.ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
self.ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

self.ds_train = self.ds_train.map(
    lambda x, y: one_hot(x, y, self.config[self.config["DATASET"]]["CLASSES"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

self.ds_train = self.ds_train.cache()
self.ds_train = self.ds_train.map(
    random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
self.ds_train = self.ds_train.map(
    random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

self.ds_train = self.ds_train.shuffle(1000)
self.ds_train = self.ds_train.batch(self.config["BATCH_SIZE"])
self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE).repeat(
    self.config["N_EPOCHS"]
)

self.ds_val = self.ds_val.map(
    lambda x, y: one_hot(x, y, self.config[self.config["DATASET"]]["CLASSES"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
self.ds_val = self.ds_val.cache()
self.ds_val = self.ds_val.batch(self.config["BATCH_SIZE"])
self.ds_val = self.ds_val.prefetch(tf.data.experimental.AUTOTUNE)


self.ds_test = self.ds_test.map(
    lambda x, y: one_hot(x, y, self.config[self.config["DATASET"]]["CLASSES"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
self.ds_test = self.ds_test.cache()
self.ds_test = self.ds_test.batch(self.config["BATCH_SIZE"])
self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)


# Data Augmentation functions


def random_flip(x, y):
    time_steps = x.shape[0]
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features // 2, 2))

        choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0, 1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features // 3, 3))

        choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0, 1.0, 1.0])
    x = tf.reshape(x, (time_steps, -1))
    return x, y


def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal(
        (time_steps, n_features), mean=0.0, stddev=0.05, dtype=tf.float64
    )
    x = x + noise
    return x, y


def one_hot(x, y, n_classes):
    return x, tf.one_hot(y, n_classes)
