import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
import math
import os
import logging
import random
import argparse
from PIL import Image

from utils.preproc_data import preprocess_seq, get_max_length
from utils.data import load_and_preprocess_data, load_and_preprocess_data_predict
from PIL import Image

import sys
import cv2

grays = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~i!lI;:,\"^.` "  # 由于控制台是白色背景，所以先密后疏/黑色背景要转置一下
gs = len(grays)  # 灰度级数

# 读入灰度图
img = cv2.imread('upload/logo1.jpg', 0)

# 宽（列）和高（行数）
w = img.shape[1]
h = img.shape[0]

# 调整长宽比（此比例为win cmd，需根据不同终端的字符长宽调整）
ratio = float(w) / h /3  # 控制台的字符长宽比

# 缩放尺度，值越小图越大
scale = w // 150  # 例如：缩小尺度

# 根据缩放长度 遍历高度 y 对应 h，x 对应 w
for y in range(0, h, int(scale * ratio)):  # 根据缩放长度 遍历高度
    for x in range(0, w, scale):  # 根据缩放长度 遍历宽度
        idx = img[y][x] * gs // 255  # 获取每个点的灰度  根据不同的灰度填写相应的替换字符
        if idx == gs:
            idx = gs - 1
        sys.stdout.write(grays[idx])  # 写入控制台
    sys.stdout.write('\n')
    sys.stdout.flush()


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.context_vector = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform',
                                              trainable=True)

    def call(self, inputs, **kwargs):
        # Compute attention scores
        attention_scores = tf.matmul(inputs, self.context_vector)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        # Apply attention weights to inputs
        weighted_inputs = tf.multiply(inputs, tf.expand_dims(attention_weights, axis=-1))
        context_vector = tf.reduce_sum(weighted_inputs, axis=1)
        return context_vector, attention_weights


def main(args):
    if args.predict:
        if not (args.load_path and args.predict_file and args.threshold and args.output_csv):
            parser.error("When using -p/--predict, --load_path, --predict_file, --threshold, and --output_csv are required.")
        import re
        def remove_x(text):
            pattern = re.compile(r"[x]*")
            return pattern.sub("", text)
        # Load the model
        model = tf.keras.models.load_model(args.load_path)

        # Get the model input shape to determine max_length
        input_shape = model.input_shape
        max_length = input_shape[1]

        # Preprocess the prediction data
        data_test = preprocess_seq(args.predict_file, max_length)
        X_val = np.array([sequence for sequence, label in data_test])
        y_val = np.array([label for sequence, label in data_test])

        # Load and preprocess the dataset
        predict_size= args.predict_size
        val_dataset = load_and_preprocess_data_predict(X_val, y_val,predict_size)
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Write CSV header
        with open(args.output_csv, 'w', newline='') as csvfile:
            import csv
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Sequence', 'Label', 'Prediction'])

        # Iterate over the validation dataset and make predictions
        for batch_inputs, raw_seq, batch_labels in val_dataset:
            batch_inputs = np.squeeze(batch_inputs, axis=1)
            batch_predictions = model.predict(batch_inputs)

            # Write predictions and sequences to CSV
            with open(args.output_csv, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for raw_sequence, prediction in zip(raw_seq.numpy(), batch_predictions):
                    decoded_sequence = raw_sequence.decode()
                    prediction_value = prediction[0]
                    label = 0 if prediction_value < args.threshold else 1
                    decoded_sequence_x = remove_x(decoded_sequence)
                    csv_row = [decoded_sequence_x, label, prediction_value]
                    csv_writer.writerow(csv_row)

        print(f"Predictions have been written to {args.output_csv}")

    if args.train:
        print("测试1")
        print("args.train")
        if not (args.train_file and args.test_file):
            print('测试2')
            parser.error("When using -t/--train, both -tf/--train_file and -ef/--test_file are required.")
        # Training mode logic goes here
        logging.basicConfig(level=logging.INFO)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

        # Use the parsed arguments to perform training
        train_file = args.train_file
        test_file = args.test_file
        output_csv = args.output_csv
        lstm_units1_values = args.lstm_units
        lstm_layers_values = args.lstm_layers
        dense_units1_values = args.dense_units
        dense_layers_values = args.dense_layers
        dropout_rate1_values = args.dropout_rate1
        dropout_layers_values = args.dropout_layers
        num_iterations = args.num_iterations
        gamma_values = args.gamma_values
        pos_weight_values = args.pos_weight_values

        random_indices = random.sample(range(num_iterations), num_iterations)

        results_df = pd.DataFrame(columns=[
            'Iteration', 'gamma', 'pos_weight', 'lstm_units1', 'dense_units1',
            'num_lstm_layers', 'dense_layers', 'dropout_rate1', 'dropout_layers', 'TP', 'FP', 'FN', 'TN', 'ACC',
            'BACC', 'Sn', 'Sp', 'MCC', 'AUC', 'Recall', 'F1_score'
        ])

        random.shuffle(random_indices)
        for i in range(args.num_iterations):
            gamma = gamma_values[i % len(gamma_values)]
            pos_weight = pos_weight_values[i % len(pos_weight_values)]
            lstm_units1 = lstm_units1_values[i % len(lstm_units1_values)]
            dense_units1 = dense_units1_values[i % len(dense_units1_values)]
            num_lstm_layers = lstm_layers_values[i % len(lstm_layers_values)]
            dense_layers = dense_layers_values[i % len(dense_layers_values)]
            dropout_rate1 = dropout_rate1_values[i % len(dropout_rate1_values)]
            dropout_layers = dropout_layers_values[i % len(dropout_layers_values)]

            train_file = args.train_file
            val_file = args.test_file
            max_length = 0
            max_length = get_max_length(train_file, val_file, max_length)
            data_train = preprocess_seq(train_file, max_length)
            data_test = preprocess_seq(val_file, max_length)

            X = np.array([sequence for sequence, label in data_train])
            y = np.array([label for sequence, label in data_train])
            X_val = np.array([sequence for sequence, label in data_test])
            y_val = np.array([label for sequence, label in data_test])

            val_dataset = load_and_preprocess_data(X_val, y_val,size = args.size)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model_path = f'model/best_model_{i+1}.h5'
            best_model = None
            best_accuracy = 0

            accuracy_list = []
            auc_list = []
            bacc_list = []
            sensitivity_list = []
            specificity_list = []
            mcc_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            batch_size = args.size
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                train_samples = len(X_train) - (len(X_train) % batch_size)
                test_samples = len(X_test) - (len(X_test) % batch_size)
                X_train = X_train[:train_samples]
                y_train = y_train[:train_samples]
                X_test = X_test[:test_samples]
                y_test = y_test[:test_samples]

                train_dataset = load_and_preprocess_data(X_train, y_train,size = args.size)
                test_dataset = load_and_preprocess_data(X_test, y_test,size = args.size)

                # 训练模型
                if args.use_fl:
                    from utils.model import train_model_fl
                    model = train_model_fl(train_dataset, test_dataset, max_length, args.size, i,
                                           lstm_units1, num_lstm_layers, dense_layers, dense_units1,
                                           dropout_layers, dropout_rate1, gamma, pos_weight)
                else:
                    from utils.model import train_model
                    model = train_model(train_dataset, test_dataset, max_length, args.size, i,
                                        lstm_units1, num_lstm_layers, dense_layers, dense_units1,
                                        dropout_layers, dropout_rate1, gamma, pos_weight)
                train_dataset, test_dataset = None, None

                batch_size = args.size
                total_samples = len(X_val)
                total_batches = (total_samples + batch_size - 1) // batch_size

                total_loss = 0.0
                total_accuracy = 0.0

                for batch_index in range(total_batches):
                    start_idx = batch_index * batch_size
                    end_idx = min((batch_index + 1) * batch_size, total_samples)

                    batch_X = X_val[start_idx:end_idx]
                    batch_y = y_val[start_idx:end_idx]
                    batch_dataset = load_and_preprocess_data(batch_X, batch_y)

                    batch_loss, batch_accuracy = model.evaluate(batch_dataset, batch_size=args.size)

                    total_loss += batch_loss * (end_idx - start_idx)
                    total_accuracy += batch_accuracy * (end_idx - start_idx)

                avg_loss = total_loss / total_samples
                avg_accuracy = total_accuracy / total_samples

                print("平均损失:", avg_loss)
                print("平均准确率:", avg_accuracy)

                if avg_accuracy > best_accuracy:
                    best_model = model
                    best_accuracy = avg_accuracy

                predicted_class = []
                y_true = []
                predicted_probabilities = []

                for batch_inputs, batch_labels in val_dataset:
                    batch_inputs = np.squeeze(batch_inputs, axis=1)
                    batch_labels = batch_labels.numpy()

                    batch_predicted_probabilities = best_model.predict_on_batch(batch_inputs)
                    threshold = 0.5
                    predicted_class.extend([1 if x[1] < threshold else 0 for x in batch_predicted_probabilities])
                    y_true.extend(batch_labels)
                    predicted_probabilities.extend(batch_predicted_probabilities)

                predicted_class = np.array(predicted_class)
                y_true = np.array(y_true)
                predicted_probabilities = np.array(predicted_probabilities)

                FN, TP, TN, FP = confusion_matrix(y_true, predicted_class).ravel()
                if np.isnan(TP) or np.isnan(FP) or np.isnan(TN) or np.isnan(FN):
                    TP = 0
                    FP = 0
                    TN = 0
                    FN = 0

                accuracy = (TP + TN) / (TP + TN + FP + FN)
                auc = roc_auc_score(y_true, predicted_probabilities[:, 1])
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                mcc = (TP * TN - FP * FN) / math.pow(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 0.5)
                bacc = 0.5 * TP / (TP + FN) + 0.5 * TN / (TN + FP)
                precision = TP / (TP + FP)
                recall = sensitivity
                f1 = 2 * precision * recall / (precision + recall)

                accuracy_list.append(accuracy)
                auc_list.append(auc)
                bacc_list.append(bacc)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
                mcc_list.append(mcc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            best_model.save(model_path)

            val_accuracy = accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_combination = {
                    'gamma': gamma,
                    'pos_weight': pos_weight,
                    'lstm_units': lstm_units1,
                    'dense_units': dense_units1,
                    'num_lstm_layers': num_lstm_layers,
                    'dense_layers': dense_layers,
                    'dropout_rate': dropout_rate1,
                }

            results_df = pd.concat([results_df, pd.DataFrame({
                'Iteration': [i],
                'gamma': [gamma],
                'pos_weight': [pos_weight],
                'lstm_units1': [lstm_units1],
                'dense_units1': [dense_units1],
                'num_lstm_layers': [num_lstm_layers],
                'dense_layers': [dense_layers],
                'dropout_rate1': [dropout_rate1],
                'dropout_layers': [dropout_layers],
                'TP': [TP],
                'FP': [FP],
                'FN': [FN],
                'TN': [TN],
                'ACC': [accuracy],
                'BACC': [bacc],
                'Sn': [sensitivity],
                'Sp': [specificity],
                'MCC': [mcc],
                'AUC': [auc],
                'Recall': [recall],
                'F1_score': [f1]
            })], ignore_index=True)

        best_combination_str = ', '.join([f'{key}: {value}' for key, value in best_combination.items()])
        logging.info('Best Combination: %s', best_combination_str)
        print(f"Best Combination: {best_combination}")

        results_df.to_csv(f'results/{args.output_csv}', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with the specified parameters.")

    group_train = parser.add_argument_group('required arguments when training')
    group_train.add_argument('-t', '--train', action='store_true', help='Training model.')
    group_train.add_argument('-tf', '--train_file', type=str, help='Path to the training data file.')
    group_train.add_argument('-ef', '--test_file', type=str, help='Path to the testing data file.')

    parser.add_argument('-s', '--size', type=int, default=8, help='Batch size.')
    parser.add_argument('-tr', '--train_result', type=str, default="./results", help='Output CSV file name.')
    parser.add_argument(
        '-lu', '--lstm_units', type=int, default=[32], nargs='+',
        help="Comma-separated list of units for LSTM layers (e.g., 32,64,128)"
    )
    parser.add_argument(
        '-ll', '--lstm_layers', type=int, default=[1], nargs='+',
        help="Number of LSTM layers (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-du', '--dense_units', type=int, default=[32], nargs='+',
        help="Comma-separated list of units for dense layers (e.g., 32,64)"
    )
    parser.add_argument(
        '-dl', '--dense_layers', type=int, default=[1], nargs='+',
        help="Number of dense layers (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-dr', '--dropout_rate1', type=float, default=[0.1], nargs='+',
        help="Comma-separated list of dropout rates (e.g., 0.1,0.2)"
    )
    parser.add_argument(
        '-dpl', '--dropout_layers', type=int, default=[0], nargs='+',
        help="Number of dropout layers (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-ni', '--num_iterations', type=int, default=20, help="Number of random parameter search iterations"
    )
    parser.add_argument(
        '-gv', '--gamma_values', type=int, default=[1], nargs='+',
        help="Comma-separated list of gamma values (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-pwv', '--pos_weight_values', type=int, default=[1], nargs='+',
        help="Comma-separated list of pos weight values (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-fl', '--use_fl', action='store_true', help="Use train_model_fl for training if specified"
    )

    group_predict = parser.add_argument_group('required arguments when predicting')
    group_predict.add_argument('-p', '--predict', action='store_true', help="Use prediction mode if specified")
    group_predict.add_argument('-lp', '--load_path', type=str, help="Path to the trained model for prediction")
    group_predict.add_argument('-pf', '--predict_file', type=str, help="Path to the file for prediction")
    group_predict.add_argument('-th', '--threshold', default=0.5, type=float,
                               help="Threshold value for classification in prediction")
    group_predict.add_argument('-o', '--output_csv', default='results/predictions.csv', type=str,
                               help="Path to the output CSV file for prediction")
    group_predict.add_argument('-ps', '--predict_size', default=64, type=int,
                               help="TXXXXXXXXXXX")
    args = parser.parse_args()
    main(args)
