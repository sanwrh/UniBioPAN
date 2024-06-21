import tensorflow as tf

from .preproc_data import preprocess_sequence
def map_fn(sequence, label):
    print(sequence)
    print(type(sequence))
    processed_sequence = tf.py_function(preprocess_sequence, [sequence], tf.float16)
    return processed_sequence, label


def load_and_preprocess_data(sequences, labels,size):
    # print(sequences)
    # print(type(sequences))
    sequences = tf.constant(sequences, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int8)

    # 创建 tf.data.Dataset，并使用 map 应用处理函数
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
    dataset = dataset.map(lambda sequence, label: map_fn(sequence, label), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.shuffle(buffer_size=len(sequences)).cache()
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(size)
    return dataset
def load_and_preprocess_data_predict(sequences, labels, batch_size):
    # 将 sequences 转换为张量
    sequences = tf.constant(sequences, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int8)
    # 创建 tf.data.Dataset，并使用 map 应用处理函数
    sequence_dataset = tf.data.Dataset.from_tensor_slices(sequences)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    # 合并数据集
    dataset = tf.data.Dataset.zip((sequence_dataset, labels_dataset))

    def map_fn(sequence, label):
        # 处理 sequence
        processed_sequence = tf.py_function(preprocess_sequence, [sequence], tf.float16)
        return processed_sequence,sequence, label

    dataset = dataset.map(lambda sequence, label: map_fn(sequence, label), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    return dataset
