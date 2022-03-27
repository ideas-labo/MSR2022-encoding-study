import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedKFold
import time
import random
from sklearn.preprocessing import MinMaxScaler
from mlp_sparse_model import MLPSparseModel # the codes for the deep learning model DeepPerf
from mlp_plain_model import MLPPlainModel # the codes for the deep learning model DeepPerf


def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    """
    This function is for running the deep learning model
    Args:
        X_train1: train input data (2/3 of the whole training data)
        Y_train1: train output data (2/3 of the whole training data)
        X_train2: validate input data (1/3 of the whole training data)
        Y_train2: validate output data (1/3 of the whole training data)
        n_layer: number of layers of the neural network
        lambd: regularized parameter

    """
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0

    # Build and train model
    model = MLPSparseModel(config)
    model.build_train()
    model.train(X_train1, Y_train1, lr_initial)

    # Evaluate trained model on validation data
    Y_pred_val = model.predict(X_train2)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))

    return abs_error, rel_error


if __name__ == "__main__":
    ####
    #### to switch between systems, comment and uncomment the following lines
    ####
    # dataset = "Data/ExaStencils.csv"
    # dataset = "Data/MongoDB.csv"
    # dataset = "Data/Trimesh.csv"
    # dataset = "Data/Irzip.csv"
    dataset = "Data/X264-DB.csv"
    ####
    #### to switch between systems, comment and uncomment the above lines
    ####



    ####
    #### change the following lines to alter the experiment settings
    ####
    random.seed(0) # set the randomseed
    N_attributes = 1 # the number of performance attributes in the csv file
    df = pd.read_csv(dataset) # read the dataset from csv file
    whole_data = df.values
    N_samples, N = whole_data.shape # get the dimensions of the dataset
    N_features = N - N_attributes
    N_train = 5000 # change this to alter the training sample size
    if N_train > N_samples:
        N_train = N_samples
    N_splits = 10 # the number of folds for the k-fold cross validation testing
    N_repeats = 5 # the number of repeats for the k-fold cross validation testing
    test_mode = False # set to False to tune the hyperparameters, which takes longer time
    save_file = False # whether to save the results
    # file name to save the results
    saving_file_name = '{}_{}.txt'.format(dataset.split('/')[1].split('.')[0],
                                          time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time())))
    if save_file:
        with open(saving_file_name, 'w') as f:  # initialize the saving file
            f.write('')
    ####
    #### change the above lines to alter the experiment settings
    ####

    print('N_samples: ', N_samples)
    print('N_train: ', N_train)
    print('N_features: ', N_features)

    # randomly collect the samples used in the experiment
    total_index = random.sample(list(range(N_samples)), N_train)

    for perf_attribute in range(1,N_attributes+1): # for each performance attribubte
        for encoding in ['Labelencoder','Scaling','Onehotencoder']: # for each encoding scheme
            ###
            ### data processing
            ###
            y = whole_data[total_index, -perf_attribute] # retrieve the performance data
            x = whole_data[total_index, 0:N_features] # retrieve the configurations
            data = [] # for the encoded configurations

            if encoding == 'Labelencoder': # for label encoding
                label_encoder = preprocessing.LabelEncoder()
                for feature in range(N_features):
                    x[:,feature] = label_encoder.fit_transform(x[:,feature])
                data = x

            elif encoding == 'Onehotencoder': # for one-hot encoding
                onehot_encoder = preprocessing.OneHotEncoder()
                x = onehot_encoder.fit_transform(x).toarray()
                data = x

            elif encoding == 'Scaling': # for scaled label encoding
                scaler = MinMaxScaler(feature_range=(0, 1))
                for feature in range(N_features):
                    x[:, feature] = scaler.fit_transform(x[:, feature].reshape(-1,1))[:,0]
                data = x

            else:
                print('Wrong encoding: ',encoding)
            ###
            ### data processing
            ###



            ###
            ### hyperparameter tuning, model training, testing and file saving
            ###
            # for each machine learning algorithm
            for algorithm in ['RandomForest', 'KNN', 'SVR', 'DecisionTree', 'LinearRegression', 'KernelRidge','Deepperf']:
                # write the experiment information
                if save_file:
                    with open(saving_file_name, 'a') as f:
                        f.write("{},{},{},{}\n".format(dataset.split('/')[1].split('.')[0], algorithm,
                                                       df.columns[-perf_attribute].replace('<', '').replace('$', ''),
                                                       encoding))

                # initialize variables
                SqError = []
                times = []
                run = 0

                # learning the ML models
                if algorithm == 'RandomForest':
                    RF = RandomForestRegressor(random_state=0)
                    # hyperparameters tu tune
                    param = {'n_estimators': np.arange(10, 100, 20),
                             'criterion': ('mse', 'mae'),
                             'max_features': ('auto', 'sqrt', 'log2'),
                             'min_samples_split': [2, 3, 4, 5, 6]
                             }
                    print("\n{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    # split the training data for k-fold cross validation testing, N_runs = n_splits * n_repeats
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats,
                                        random_state=1)

                    for train_index, test_index in rkf.split(data): # for each run
                        start_time = time.time() # initialize time
                        run += 1
                        print('Run:', run)
                        # tune the parameters using k-fold cv grid search
                        gridS = GridSearchCV(RF, param)
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(RandomForestRegressor(**gridS.best_params_,random_state=0), data[test_index],
                                                   y[test_index], cv=5)  ###cv decides the k for k-fold grid search
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred)) # compute the error rate
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe) # save the RMSE
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time) # save the training time
                        # save the results
                        if save_file:
                            with open(saving_file_name, 'a') as f:
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                ### the following lines are the same as the above, so the comments are omitted
                elif algorithm == 'KNN':
                    knn = KNeighborsRegressor()
                    param = {'n_neighbors': [3, 5, 7, 10, 11, 15],
                             'weights': ('uniform', 'distance'),
                             'algorithm': ['auto'],  # 'ball_tree','kd_tree'),
                             'leaf_size': [10, 20, 30, 40, 50, 60, 70, 80, 90],
                             }
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        gridS = GridSearchCV(knn, param)  ###
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(KNeighborsRegressor(**gridS.best_params_), data[test_index],
                                                   y[test_index],
                                                   cv=5)  ###
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                elif algorithm == 'SVR':
                    values = np.arange(0.01, 5, 0.5)
                    svr = SVR()
                    param = {'kernel': ('linear', 'rbf'),
                             'degree': [2, 3, 4, 5],
                             'gamma': ('scale', 'auto'),
                             'coef0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             'epsilon': [0.01, 1]
                             }
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        gridS = GridSearchCV(svr, param)  ###
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(SVR(**gridS.best_params_), data[test_index], y[test_index],
                                                   cv=5)  ###
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                elif algorithm == 'DecisionTree':
                    DT = DecisionTreeRegressor(random_state=0)
                    param = {'criterion': ('mse', 'friedman_mse', 'mae'),
                             'splitter': ('best', 'random'),
                             'min_samples_split': [0.1, 0.2, 0.5, 1.0, 2, 3, 4, 5, 6, 7]
                             }
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        gridS = GridSearchCV(DT, param)  ###
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(DecisionTreeRegressor(**gridS.best_params_,random_state=0), data[test_index],
                                                   y[test_index],
                                                   cv=5)  ###
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                elif algorithm == 'LinearRegression':
                    LR = LinearRegression()
                    param = {'fit_intercept': ('True', 'False'),
                             'normalize': ('True', 'False'),
                             'n_jobs': [1, -1]
                             }
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        gridS = GridSearchCV(LR, param)  ###
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(LinearRegression(**gridS.best_params_), data[test_index],
                                                   y[test_index],
                                                   cv=5)  ###
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                elif algorithm == 'KernelRidge':
                    x1 = np.arange(0.1, 5, 0.5)
                    KR = KernelRidge()
                    param = {'alpha': x1,
                             'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                             'coef0': [1, 2, 3, 4, 5]
                             }
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        gridS = GridSearchCV(KR, param)  ###
                        gridS.fit(data[train_index], y[train_index])
                        y_pred = cross_val_predict(KernelRidge(**gridS.best_params_), data[test_index], y[test_index],
                                                   cv=5)  ###
                        sqe = math.sqrt(mean_squared_error(y[test_index], y_pred))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                ### Deepperf is a STOA deep learning model for performance prediction
                elif algorithm == 'Deepperf':
                    print("{},{},{},{}".format(dataset.split('/')[1].split('.')[0], algorithm,
                                               df.columns[-perf_attribute].replace('<', '').replace('$', ''), encoding))
                    rkf = RepeatedKFold(n_splits=N_splits, n_repeats=N_repeats, random_state=1)
                    for train_index, test_index in rkf.split(data):
                        run += 1
                        print('Run:', run)
                        start_time = time.time()
                        N_train = len(train_index)
                        N_test = len(test_index)

                        # retrieve the training and testing data
                        X_train_deepperf = data[train_index]
                        Y_train_deepperf = y[train_index][:, np.newaxis]
                        X_test_deepperf = data[test_index]
                        Y_test_deepperf = y[test_index][:, np.newaxis]

                        # Scale X and Y
                        temp_max_X = np.amax(X_train_deepperf, axis=0)
                        if 0 in temp_max_X:
                            temp_max_X[temp_max_X == 0] = 1
                        X_train_deepperf = np.divide(X_train_deepperf, temp_max_X)
                        temp_max_Y = np.max(Y_train_deepperf) / 100
                        if temp_max_Y == 0:
                            temp_max_Y = 1
                        Y_train_deepperf = np.divide(Y_train_deepperf, temp_max_Y)
                        X_test_deepperf = np.divide(X_test_deepperf, temp_max_X)

                        # Split train data into 2 parts (67-33)
                        N_cross = int(np.ceil(len(X_train_deepperf) * 2 / 3))
                        X_train1 = X_train_deepperf[0:N_cross, :]
                        Y_train1 = Y_train_deepperf[0:N_cross]
                        X_train2 = X_train_deepperf[N_cross:N_train, :]
                        Y_train2 = Y_train_deepperf[N_cross:N_train]

                        if test_mode == True:
                            # ---default settings, just for testing---
                            lr_opt = 0.123
                            n_layer_opt = 3
                            lambda_f = 0.123
                            config = dict()
                            config['num_neuron'] = 128
                            config['num_input'] = len(X_train1[0])
                            config['num_layer'] = n_layer_opt
                            config['lambda'] = lambda_f
                            config['verbose'] = 0

                        if test_mode == False:
                            # hyperparameter tuning
                            print('Tuning hyperparameters for the neural network ...')
                            print('Step 1: Tuning the number of layers and the learning rate ...')
                            config = dict()
                            config['num_input'] = len(X_train1[0])
                            config['num_neuron'] = 128
                            config['lambda'] = 'NA'
                            config['decay'] = 'NA'
                            config['verbose'] = 0
                            dir_output = 'C:/Users/Downloads'
                            abs_error_all = np.zeros((15, 4))
                            abs_error_all_train = np.zeros((15, 4))
                            abs_error_layer_lr = np.zeros((15, 2))
                            abs_err_layer_lr_min = 100
                            count = 0
                            layer_range = range(2, 15)
                            lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
                            for n_layer in layer_range:
                                config['num_layer'] = n_layer
                                for lr_index, lr_initial in enumerate(lr_range):
                                    model = MLPPlainModel(config, dir_output)
                                    model.build_train()
                                    model.train(X_train1, Y_train1, lr_initial)

                                    Y_pred_train = model.predict(X_train1)
                                    abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                    Y_pred_val = model.predict(X_train2)
                                    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                                    abs_error_all[int(n_layer), lr_index] = abs_error

                                # Pick the learning rate that has the smallest train cost
                                # Save testing abs_error correspond to the chosen learning_rate
                                temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                temp_idx = np.where(abs(temp) < 0.0001)[0]
                                if len(temp_idx) > 0:
                                    lr_best = lr_range[np.max(temp_idx)]
                                    err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                                else:
                                    lr_best = lr_range[np.argmin(temp)]
                                    err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                                abs_error_layer_lr[int(n_layer), 0] = err_val_best
                                abs_error_layer_lr[int(n_layer), 1] = lr_best

                                if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                                    abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                                         np.argmin(temp)]
                                    count = 0
                                else:
                                    count += 1

                                if count >= 2:
                                    break
                            abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

                            # Get the optimal number of layers
                            n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 5

                            # Find the optimal learning rate of the specific layer
                            config['num_layer'] = n_layer_opt
                            for lr_index, lr_initial in enumerate(lr_range):
                                model = MLPPlainModel(config, dir_output)
                                model.build_train()
                                model.train(X_train1, Y_train1, lr_initial)

                                Y_pred_train = model.predict(X_train1)
                                abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                Y_pred_val = model.predict(X_train2)
                                abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                                abs_error_all[int(n_layer), lr_index] = abs_error

                            temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                            temp_idx = np.where(abs(temp) < 0.0001)[0]
                            if len(temp_idx) > 0:
                                lr_best = lr_range[np.max(temp_idx)]
                            else:
                                lr_best = lr_range[np.argmin(temp)]

                            lr_opt = lr_best
                            print('The optimal number of layers: {}'.format(n_layer_opt))
                            print('The optimal learning rate: {:.4f}'.format(lr_opt))

                            print('Step 2: Tuning the l1 regularized hyperparameter ...')
                            # Use grid search to find the right value of lambda
                            lambda_range = np.logspace(-2, np.log10(1000), 30)
                            error_min = np.zeros((1, len(lambda_range)))
                            rel_error_min = np.zeros((1, len(lambda_range)))
                            decay = 'NA'
                            for idx, lambd in enumerate(lambda_range):
                                val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                                                       X_train2, Y_train2,
                                                                       n_layer_opt, lambd, lr_opt)
                                error_min[0, idx] = val_abserror
                                rel_error_min[0, idx] = val_relerror

                            # Find the value of lambda that minimize error_min
                            lambda_f = lambda_range[np.argmin(error_min)]
                            print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

                            # Solve the final NN with the chosen lambda_f on the training data
                            config = dict()
                            config['num_neuron'] = 128
                            config['num_input'] = X_train_deepperf.shape[1]
                            config['num_layer'] = n_layer_opt
                            config['lambda'] = lambda_f
                            config['verbose'] = 0
                            # # ---for testing, stop comment here---

                        # train deepperf model
                        deepperf_model = MLPSparseModel(config)
                        deepperf_model.build_train()
                        deepperf_model.train(X_train_deepperf, Y_train_deepperf, lr_opt)

                        # test result
                        print('Testing...')
                        Y_pred_test = deepperf_model.predict(X_test_deepperf)
                        Y1_pred_test = temp_max_Y * Y_pred_test[:, 0:1]

                        sqe = math.sqrt(mean_squared_error(Y_test_deepperf.ravel(), Y1_pred_test.ravel()))
                        print('RMSE:', round(sqe, 2))
                        SqError.append(sqe)
                        print('Time(s):', round((time.time() - start_time), 2))
                        times.append(time.time() - start_time)

                        # save the results
                        if save_file:
                            with open(saving_file_name, 'a') as f:
                                f.write('RMSE:{}\n'.format(sqe))
                                f.write('Time:{}\n'.format(time.time() - start_time))

                else:
                    print('Wrong algorithm: ',algorithm)