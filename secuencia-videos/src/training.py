import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import autokeras as ak
import scikitplot as skplt
import pickle5 as pickle
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam




st.set_option('deprecation.showPyplotGlobalUse', False)

# AutoKeras
def autokeras_model(csv,namemodel,testsize):
     # Genera modelo en base al csv generado
    try:
        # Cargar datos
        df = pd.read_csv(csv, header=0)

        st.dataframe(df.head())
        dataset = df.values

        X = dataset[:, 0:50].astype(float)
        Y = dataset[:, 50]

        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print(class_mapping)
        # Train/Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, encoder_Y, test_size=testsize, random_state=9)
        clf3 = ak.StructuredDataClassifier(max_trials=1)
        clf3.fit(X_train, Y_train, epochs=25)

        # Test modelo
        y_pred_autok3 = clf3.predict(X_test)
        accuracy_autok3_df = metrics.accuracy_score(Y_test, y_pred_autok3)

        # Evaluate the best model with testing data.
        st.text("Accuracy Evaluate: {accuracy}".format(accuracy=clf3.evaluate(X_test, Y_test)))
        st.text("Accuracy Score: {accuracy}".format(accuracy=accuracy_autok3_df))


        # Matriz de confusion 
        # print(confusion_matrix(Y_test, y_pred_autok3))
        skplt.metrics.plot_confusion_matrix(Y_test,y_pred_autok3,figsize=(12,12),normalize=True)
        st.pyplot()
        
        # Exportar Modelo
        st.text("Resumen modelo")
        model = clf3.export_model()

        # Obtener mejor modelo 
        best_model = clf3.tuner.get_best_model()

        #Save the model
        try:
            best_model.save("output/model/automl_"+namemodel+".h5")
        except Exception as e:
            st.exception("Save AutoML->"+e)
        del model
        del best_model
        del clf3
    except Exception as e:
        st.exception(e)

# Random Forest
def randomForestModel(csv,namemodel,testsize):
    # Genera modelo en base al csv generado
    try:
        # Cargar datos
        df = pd.read_csv(csv, header=0)

        st.dataframe(df.head())
        dataset = df.values

        X = dataset[:, 0:50].astype(float)
        Y = dataset[:, 50]

        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        
        # Train/Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, encoder_Y, test_size=testsize, random_state=9)

        # Construir modelo con sklearn
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        model.fit(X_train, Y_train)

        y_pred_forest = model.predict(X_test)
        accuracy_forest = metrics.accuracy_score(Y_test, y_pred_forest)
        st.text("Accuracy: "+str(accuracy_forest))

        disp = metrics.plot_confusion_matrix(model,X_test,Y_test,
            display_labels=['caminar','saltar'],
            cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title("Random Forest confusion matrix")
        st.pyplot()


        st.text(classification_report(Y_test,y_pred_forest))

        # Save the model
        pkl_filename = "C:/TFG/openpose/build_windows/examples/user_code/output/model/tree_"+namemodel+".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        st.exception(e)


def mlpModel(csv,namemodel,testsize):
    try:
        # Cargar datos
        df = pd.read_csv(csv, header=0)

        st.dataframe(df.head())
        dataset = df.values

        X = dataset[:, 0:50].astype(float)
        Y = dataset[:, 50]

        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        
        # Train/Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, encoder_Y, test_size=testsize, random_state=9)

        mlp = MLPClassifier(hidden_layer_sizes=[100,100], alpha = 5, solver = 'adam', max_iter = 5000, epsilon=1e-08)

        mlp_model = mlp.fit(X_train,Y_train)

        y_pred_mlp = mlp_model.predict(X_test)
        accuracy_mlp_df = metrics.accuracy_score(Y_test, y_pred_mlp)

        st.text("Accuracy: "+str(accuracy_mlp_df))

        disp = metrics.plot_confusion_matrix(mlp_model,X_test,Y_test,
            display_labels=['caminar','saltar','carrera','batida'],
            cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title("MLP confusion matrix")
        st.pyplot()

        st.text(classification_report(Y_test,y_pred_mlp))

        # Save the model
        pkl_filename = "C:/TFG/openpose/build_windows/examples/user_code/output/model/mlp_"+namemodel+".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(mlp_model, file)
    except Exception as e:
        st.exception(e)

def gpcModel(csv,namemodel,testsize):
    try:
        # Cargar datos
        df = pd.read_csv(csv, header=0)

        st.dataframe(df.head())
        dataset = df.values

        X = dataset[:, 0:50].astype(float)
        Y = dataset[:, 50]

        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        
        # Train/Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, encoder_Y, test_size=testsize, random_state=9)

        kernel = 1.0 * RBF(1.0)
        gpc = GaussianProcessClassifier(kernel, max_iter_predict=5000, n_jobs=-1)

        gpc_model = gpc.fit(X_train,Y_train)

        y_pred_gpc = gpc_model.predict(X_test)
        accuracy_gpc_df = metrics.accuracy_score(Y_test, y_pred_gpc)

        st.text("Accuracy: "+str(accuracy_gpc_df))

        disp = metrics.plot_confusion_matrix(gpc_model,X_test,Y_test,
            display_labels=['caminar','saltar','carrera','batida'],
            cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title("GPC confusion matrix")
        st.pyplot()

        st.text(classification_report(Y_test,y_pred_gpc))

        # Save the model
        pkl_filename = "C:/TFG/openpose/build_windows/examples/user_code/output/model/gpc_"+namemodel+".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(gpc_model, file)
    except Exception as e:
        st.exception(e)

def LSTMModel(csv,namemodel,testsize):
    # Genera modelo en base al csv generado
    try:
        # Cargar datos
        df = pd.read_csv(csv, header=0)

        st.dataframe(df.head())
        dataset = df.values

        X = dataset[:, 0:50].astype(float)
        Y = dataset[:, 50]

        encoder = LabelEncoder()
        encoder_Y = encoder.fit_transform(Y)
        class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        
        # Train/Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, encoder_Y, test_size=testsize, random_state=9)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50,dropout=0.2, recurrent_dropout=0.2,input_shape=(50,1)))
        model.add(Dense(50,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1,activation='softmax'))


        # Training
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
        accr = model.evaluate(X_test,Y_test)
        st.text('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        st.text(model.summary())
        history = model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, Y_test))
        # st.text(lstm_model.summary())

        
        # Save the model
        try:
            model.save("output/model/lstm_"+namemodel+".h5")
        except Exception as e:
            st.exception(e)
        del model
    except Exception as e:
        st.exception(e)
