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


from utils.preproc_data import preprocess_seq, get_max_length, read_sequences
from utils.data import load_and_preprocess_data, load_and_preprocess_data_predict


import sys
import cv2

grays = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~i!lI;:,\"^.` "
gs = len(grays)  # Number of gray levels

# Read in the grayscale image
img = cv2.imread('upload/logo1.jpg', 0)

# Width (columns) and height (rows)
w = img.shape[1]
h = img.shape[0]

# Adjust the aspect ratio (this ratio is for the Windows CMD, it may need adjustment for different terminal character aspect ratios)
ratio = float(w) / h /3  # Console character aspect ratio

# Scaling factor, the smaller the value, the larger the image
scale = max(w // 100, 2)  # reduce the scale

# Traverse the height y corresponding to h, x corresponding to w, based on the scaling factor
for y in range(0, h, int(scale * ratio)):
    for x in range(0, w, scale):
        idx = img[y][x] * gs // 255
        if idx == gs:
            idx = gs - 1
        sys.stdout.write(grays[idx])
    sys.stdout.write('\n')
    sys.stdout.flush()
welcome_message = """
***********************************************
****                                       ****
****           Welcome to UniBioPAN        ****
****                                       ****
***********************************************
UniBioPAN: A Comprehensive Tool for Bioactive Peptide Classification

"""
lines = welcome_message.split('\n')
max_length = max(len(line) for line in lines)
centered_lines = [line.center(max_length) for line in lines]
centered_message = '\n'.join(centered_lines)
print(welcome_message)
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
        if not (args.load_path and args.predict_file):
            parser.error("When using -p/--predict and --load_path are required.")
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

        filename = args.predict_file
        sequences = read_sequences(filename)
        preprocess_predictions = preprocess_seq(sequences, max_length)
        from utils.preproc_data import preprocess_sequence
        batch_frame = np.vstack([preprocess_sequence(seq) for seq in preprocess_predictions])


        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Predictions
        # Predictions in batches
        num_samples = len(batch_frame)
        predictions_list = []

        for i in range(0, num_samples, args.predict_size):
            batch_data = batch_frame[i:i + args.predict_size]
            batch_predictions = model.predict(batch_data)
            predictions_list.append(batch_predictions)
        batch_predictions = np.vstack(predictions_list)
        # Write predictions to CSV file
        with open(output_dir, 'w', newline='') as csvfile:
            import csv
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Sequence", "Label", "Prediction"])

            for seq, predictions in zip(sequences, batch_predictions):
                seq_str = f'{seq}'
                if len(seq) > max_length:
                    label = 'out of max length'
                    predictions_str = 'out of max length'
                else:
                    label = 0 if predictions < args.threshold else 1
                    prediction_value = predictions[0]
                    predictions_str = f"{prediction_value:.6f}"

                csv_writer.writerow([seq_str, label, predictions_str])
        print(f"Predictions have been written to {args.output_csv}")

    elif args.train:
        if not (args.train_file and args.test_file):

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

                # training model
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

                print("average loss:", avg_loss)
                print("average accuracy:", avg_accuracy)

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
    parser = argparse.ArgumentParser(description="Training Arguments (required when --train is specified):")

    group_train = parser.add_argument_group('required arguments when training')
    group_train.add_argument('-t', '--train', action='store_true', help='Activate training mode.')
    group_train.add_argument('-tf', '--train_file', type=str, help='Path to the training data file.')
    group_train.add_argument('-ef', '--test_file', type=str, help='Path to the testing data file.')

    parser.add_argument('-s', '--size', type=int, default=8, help='Specify the batch size for training.')
    parser.add_argument('-tr', '--train_result', type=str, default="./results", help='Set the output filename for training results (CSV format).')
    parser.add_argument(
        '-lu', '--lstm_units', type=int, default=[32], nargs='+',
        help="Define the number of units for each LSTM layer (e.g., 32, 64, 96)."
    )
    parser.add_argument(
        '-ll', '--lstm_layers', type=int, default=[1], nargs='+',
        help="Define the number of LSTM layers in the network (e.g., 1, 2, 3)."
    )
    parser.add_argument(
        '-du', '--dense_units', type=int, default=[32], nargs='+',
        help="Define the number of units for each dense layer (e.g., 32, 64,96)."
    )
    parser.add_argument(
        '-dl', '--dense_layers', type=int, default=[1], nargs='+',
        help="Define the number of dense layers in the network (e.g., 1, 2, 3)."
    )
    parser.add_argument(
        '-dr', '--dropout_rate1', type=float, default=[0.1], nargs='+',
        help="Define the number of dropout layers in the network (e.g., 1, 2, 3)."
    )
    parser.add_argument(
        '-dpl', '--dropout_layers', type=int, default=[0], nargs='+',
        help="Define the number of dropout layers (e.g., 1,2,3)"
    )
    parser.add_argument(
        '-ni', '--num_iterations', type=int, default=20, help="Define the number of iterations for random hyperparameter search."
    )
    parser.add_argument(
        '-gv', '--gamma_values', type=int, default=[1], nargs='+',
        help="Define a list of gamma values to explore during hyperparameter tuning. Typically between 1.0 and 5.0, increase the focus on samples that are more difficult to classify."
    )
    parser.add_argument(
        '-pwv', '--pos_weight_values', type=int, default=[1], nargs='+',
        help="Define a list of positive class weight values for imbalanced datasets. "
    )
    parser.add_argument(
        '-fl', '--use_fl', action='store_true', help=" Enable the use of focal loss function during training. Values less than 1.0 emphasize positive samples in imbalanced datasets where positive samples are rare."
    )

    group_predict = parser.add_argument_group('Prediction Arguments (required when --predict is specified):')
    group_predict.add_argument('-p', '--predict', action='store_true', help="Activate prediction mode.")
    group_predict.add_argument('-lp', '--load_path', type=str, help="Path to the trained model file for prediction.")
    group_predict.add_argument('-pf', '--predict_file', type=str, help="Path to the data file containing sequences for prediction.")
    group_predict.add_argument('-th', '--threshold', default=0.5, type=float,
                               help="Set the classification threshold for prediction results.")
    group_predict.add_argument('-o', '--output_csv', default='results/predictions.csv', type=str,
                               help="Set the output filename for prediction results (CSV format).")
    group_predict.add_argument('-ps', '--predict_size', default=64, type=int,
                               help="Set the size of predict")#####
    args = parser.parse_args()
    main(args)
