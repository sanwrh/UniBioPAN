import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
# from memory_profiler import profile
# @profile
def preprocess_sequence(sequence):
    # 读取所有照片并将它们存储在一个字典中
    images = {}
    folder_path = "residues32/IA"
    file_names = os.listdir(folder_path)

    # 加载和预处理图像
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.resize(image, (32, 32))  # 调整图像大小为 32x32
        image = tf.cast(image, tf.float16) / 255.0  # 转换为 float32 并标准化到 [0, 1] 范围内
        # # 应用 Z-分数标准化（减去均值并除以标准差）
        # mean = tf.math.reduce_mean(image)
        # std = tf.math.reduce_std(image)
        # image = (image - mean) / std
        # 将NaN替换为0
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
        images[file_path[14:-4]] = image

    def map_seq(input_str):
        # print(input_str)
        # # 查看图像的形状以调试
        char_images = []  # 创建一个空列表
        prev_index = None
        # 遍历输入字符串中的每个字符
        for index, char in enumerate(input_str):
            # 如果当前字符是 "x"，记住它的索引并退出循环
            if char == 'x':
                prev_index = index-1
                break
        # 如果没有找到 "x"，则记住最后一个字符的索引
        if prev_index is None and len(input_str) > 0:
            prev_index = len(input_str) - 1

        # 遍历输入字符串中的每个字符
        for n in range(len(input_str)):
            # for char in input_str[n:n+1]:
            if n == prev_index:
                char = input_str[n]
                image_key = char + '_C'
                #print(image_key)
                char_tensor = tf.convert_to_tensor(images.get(image_key))
                char_images.append(char_tensor)
            elif n == 0:
                char = input_str[n]
                image_key = char + '_N'
                char_tensor = tf.convert_to_tensor(images.get(image_key))
                char_images.append(char_tensor)
            # 检查字符是否在images字典中
            elif n != prev_index:
                char = input_str[n]
                #print(char)
                # 如果在images字典中，将对应的图像转换为Tensor并添加到列表中
                char_tensor = tf.convert_to_tensor(images.get(char))
                char_images.append(char_tensor)
        char_images = np.array(char_images)

        seq_frames = tf.stack(char_images, axis=0)
        # del char_images,char,char_tensor,input_str
        return seq_frames
    # print('sequence的类型')
    # print(type(sequence))
    input_seq = sequence.numpy().decode("utf-8")
    #print("input sequence"+input_seq)
    processed_data = []
    # for seq in input_seq:
    #     print(seq)
    seq_frames = map_seq(input_seq)
    processed_data.append(seq_frames)
    processed_data = tf.convert_to_tensor(processed_data)
    #processed_data = tf.squeeze(processed_data, axis=1)
    # del images, file_name,file_path, image, images ,seq_frames
    return processed_data

# def preprocess_seq(filename, max_length):
#     data = pd.read_excel(filename, engine='openpyxl', keep_default_na=False, na_values=[''])
#     sequences = data['sequence'].tolist()
#     labels = data['label'].tolist()
#     processed_data = []
#     for seq, label in zip(sequences, labels):
#         # 移除行尾空格并填充到指定长度
#         seq = seq.strip().ljust(max_length, 'x')
#         processed_data.append((seq, label))  # 将数据和标签打包成一个元组并添加到列表中
#     return processed_data
import os
import pandas as pd
from Bio import SeqIO  # 需要安装 biopython

def preprocess_seq(filename, max_length):
    _, ext = os.path.splitext(filename)
    processed_data = []

    if ext == ".fasta":
        with open(filename, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_id = record.id.split('|')[0]  # 提取序列ID，如 seq1
                label = int(record.id.split('|')[1])  # 提取标签，如 0 或 1
                sequence = str(record.seq).strip().ljust(max_length, 'x')
                processed_data.append((sequence, label))
    elif ext == ".xlsx":
        data = pd.read_excel(filename, engine='openpyxl', keep_default_na=False, na_values=[''])
        sequences = list(zip(data['sequence'].tolist(), data['label'].tolist()))
        for seq, label in sequences:
            seq = seq.strip().ljust(max_length, 'x')
            processed_data.append((seq, label))
    elif ext == ".csv":
        data = pd.read_csv(filename, na_filter=False)
        sequences = list(zip(data['sequence'].tolist(), data['label'].tolist()))
        for seq, label in sequences:
            seq = seq.strip().ljust(max_length, 'x')
            processed_data.append((seq, label))
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

    return processed_data

def get_max_length(filename1,filename2,max_length):
    def count_max_length(data):
        sequences = data['sequence'].tolist()
        labels = data['label'].tolist()
        max_length = 0
        positive_sequences = []
        negative_sequences = []
        for seq, label in zip(sequences, labels):
            if label == 1:
                positive_sequences.append(seq)
            else:
                negative_sequences.append(seq)
            max_length = max(max_length, len(seq))
        del positive_sequences, negative_sequences
        return max_length
    # 从文件中读取每一行并将其与相应的照片相关联
    data1 = pd.read_excel(filename1, engine='openpyxl', keep_default_na=False, na_values=[''])  # 指定engine为'openpyxl'或'xlrd'
    data2 = pd.read_excel(filename2, engine='openpyxl', keep_default_na=False, na_values=[''])
    max_length1 = count_max_length(data1)
    max_length2 = count_max_length(data2)
    if max_length1>max_length:
        max_length =max_length1
    if max_length2>max_length1:
        max_length=max_length2
    print("数据集中字符最大长度:", max_length)
    del data2,data1,max_length1,max_length2
    return max_length
