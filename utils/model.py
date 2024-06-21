#creat_model
import tensorflow as tf
import math
from focal_loss import BinaryFocalLoss
from tensorflow.python.keras.callbacks import EarlyStopping


# 创建一个回调函数，用于在每个epoch结束时打印相关信息
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1} - Train Loss: {logs['loss']:.4f}, Train Accuracy: {logs['accuracy']:.4f}, Test Loss: {logs['val_loss']:.4f}, Test Accuracy: {logs['val_accuracy']:.4f}")


# 定义学习率回调函数，用于在每个epoch结束时打印学习率
class LearningRateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        print(f"Epoch {epoch + 1} - Learning Rate: {lr:.6f}")

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.6
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def create_model(max_length, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1):
    learning_rate = 0.0001
    momentum = 0.5
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)
    # 输入序列的长度
    sequence_length = max_length
    input_shape = (sequence_length, 32, 32, 3)
    inputs = tf.keras.Input(shape=input_shape)
    # model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same'))(
    # initializer = tf.keras.initializers.HeNormal(seed=123456)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(
        inputs)
    for _ in range(2):
        model = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(
            model)
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))(model)
    for _ in range(1):
        model = tf.keras.layers.BatchNormalization()(model)  # Adding a normalization layer
    model = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(model)
    for _ in range(2):
        model = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(model)
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))(model)
    for _ in range(1):
        model = tf.keras.layers.BatchNormalization()(model)  # Adding a normalization layer

    model = tf.keras.layers.Reshape((sequence_length, -1))(model)
    lstm_units = lstm_units1
    # 根据num_lstm_layers创建指定层数的LSTM层
    for _ in range(num_lstm_layers):
        model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units1, return_sequences=True))(model)
    # 添加 Temporal Attention 机制
    model = tf.keras.layers.Reshape((-1, lstm_units * 2))(model)
    permute1 = tf.keras.layers.Permute((2, 1))(model)
    attention_probs = tf.keras.layers.Dense(units=1, activation='softmax')(permute1)
    permute2 = tf.keras.layers.Permute((2, 1))(attention_probs)
    model = tf.keras.layers.Multiply()([model, permute2])
    model = tf.keras.layers.Flatten()(model)
    # 添加全连接层和dropout层
    for _ in range(dense_layers):
        model = tf.keras.layers.Dense(units=dense_units1)(model)
    for _ in range(dropout_layers):
        model = tf.keras.layers.Dropout(rate=dropout_rate1)(model)
    num_classes = 2
    outputs = tf.keras.layers.Dense(units=num_classes, activation='sigmoid')(model)
    # 创建模型
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=momentum, decay=0.0, nesterov=False)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 编译模型
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=adam, loss=BinaryFocalLoss(gamma=gamma,pos_weight=pos_weight), metrics=['accuracy'])  # rmsprop adam
    # 打印模型结构
    model.summary()
    return model

def create_model_fl(max_length, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1, gamma, pos_weight):
    learning_rate = 0.0001
    momentum = 0.5
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)
    # 输入序列的长度
    sequence_length = max_length
    input_shape = (sequence_length, 32, 32, 3)
    inputs = tf.keras.Input(shape=input_shape)
    # model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same'))(
    # initializer = tf.keras.initializers.HeNormal(seed=123456)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(
        inputs)
    for _ in range(2):
        model = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(
            model)
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))(model)
    for _ in range(1):
        model = tf.keras.layers.BatchNormalization()(model)  # Adding a normalization layer
    model = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(model)
    for _ in range(2):
        model = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(model)
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))(model)
    for _ in range(1):
        model = tf.keras.layers.BatchNormalization()(model)  # Adding a normalization layer

    model = tf.keras.layers.Reshape((sequence_length, -1))(model)
    lstm_units = lstm_units1
    # 根据num_lstm_layers创建指定层数的LSTM层
    for _ in range(num_lstm_layers):
        model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units1, return_sequences=True))(model)
    # 添加 Temporal Attention 机制
    model = tf.keras.layers.Reshape((-1, lstm_units * 2))(model)
    permute1 = tf.keras.layers.Permute((2, 1))(model)
    attention_probs = tf.keras.layers.Dense(units=1, activation='softmax')(permute1)
    permute2 = tf.keras.layers.Permute((2, 1))(attention_probs)
    model = tf.keras.layers.Multiply()([model, permute2])
    model = tf.keras.layers.Flatten()(model)
    # 添加全连接层和dropout层
    for _ in range(dense_layers):
        model = tf.keras.layers.Dense(units=dense_units1)(model)
    for _ in range(dropout_layers):
        model = tf.keras.layers.Dropout(rate=dropout_rate1)(model)
    num_classes = 2
    outputs = tf.keras.layers.Dense(units=num_classes, activation='sigmoid')(model)
    # 创建模型
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=momentum, decay=0.0, nesterov=False)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 编译模型
    #model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss=BinaryFocalLoss(gamma=gamma,pos_weight=pos_weight), metrics=['accuracy'])  # rmsprop adam
    # 打印模型结构
    model.summary()
    return model

def train_model(train_dataset, test_dataset, max_length, size, i, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1, gamma, pos_weight):
    strategy = tf.distribute.MirroredStrategy(devices=['GPU:0', 'GPU:1'])
    with strategy.scope():
        # Define ModelCheckpoint callback
        print(i)
        model_path = 'best_model' + str(i + 1) + '.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True,
                                                        mode='max', verbose=0)
        # 定义 EarlyStopping 回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                         min_delta=1e-4, mode='min')
        learning_rate = 0.0001
        model = create_model(max_length, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1)
        model.optimizer.lr.assign(learning_rate)
        callbacks = [reduce_lr,early_stopping,checkpoint]
        model.fit(train_dataset, epochs=400, batch_size=size, validation_data=test_dataset,
                  callbacks=callbacks, verbose=2)
    return model


def train_model_fl(train_dataset, test_dataset, max_length, size, i, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1, gamma, pos_weight):
    strategy = tf.distribute.MirroredStrategy(devices=['GPU:0', 'GPU:1'])
    with strategy.scope():
        # Define ModelCheckpoint callback
        print(i)
        model_path = 'best_model' + str(i + 1) + '.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True,
                                                        mode='max', verbose=0)
        # 定义 EarlyStopping 回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                         min_delta=1e-4, mode='min')
        learning_rate = 0.0001
        model = create_model_fl(max_length, lstm_units1, num_lstm_layers, dense_layers, dense_units1, dropout_layers, dropout_rate1,  gamma, pos_weight)
        model.optimizer.lr.assign(learning_rate)
        callbacks = [reduce_lr,early_stopping,checkpoint]
        model.fit(train_dataset, epochs=400, batch_size=size, validation_data=test_dataset,
                  callbacks=callbacks, verbose=2)
    return model