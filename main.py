from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from keras.models import Sequential
from keras.layers import *

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def decisionTree(imputar_datos, normalizar_datos, scaler_type, rebalancear, oversampling, undersampling,
                 smote_sampling, under_sampling_ratio, features):
    # Leemos los archivos
    print("\nLeyendo archivos...")
    train = pd.read_csv("features_2/training.csv")
    validation = pd.read_csv("features_2/validation.csv")
    test = pd.read_csv("features_2/test.csv")
    random_state = np.random.randint(1000)

    X_train = train.loc[:, features]
    y_train = train.MSG_CLICKED
    X_validation = validation.loc[:, features]
    y_validation = validation.MSG_CLICKED
    X_test = test.loc[:, features]
    y_test = test.MSG_CLICKED

    if imputar_datos:
        print("Imputando datos...")
        # Por simpleza solo vamos a llenar los datos con el valor promedio
        # probablemente podemos hacer algo mejor...
        imputer = SimpleImputer().fit(X_train)
        X_train = imputer.transform(X_train)
        X_validation = imputer.transform(X_validation)
        X_test = imputer.transform(X_test)

    if normalizar_datos:
        print("Estandarizando datos...")
        if scaler_type == "Standard":
            scaler = StandardScaler().fit(X_train)
        else:
            scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)

    if rebalancear:
        print("Rebalanceando...")
        if oversampling:
            sm = SMOTE(random_state=random_state, sampling_strategy=smote_sampling)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        if undersampling:
            under = RandomUnderSampler(random_state=random_state, sampling_strategy=under_sampling_ratio)
            X_train, y_train = under.fit_resample(X_train, y_train)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for validation dataset
    y_val = clf.predict(X_validation)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy Validation:", metrics.accuracy_score(y_validation, y_val))
    print("Precision Validation:", metrics.precision_score(y_validation, y_val))
    print("Recall Validation:", metrics.recall_score(y_validation, y_val))
    print("F1 Validation:", metrics.f1_score(y_validation, y_val))

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy Test:", metrics.accuracy_score(y_test, y_pred))
    print("Precision Test:", metrics.precision_score(y_test, y_pred))
    print("Recall Test:", metrics.recall_score(y_test, y_pred))
    print("F1 Test:", metrics.f1_score(y_test, y_pred))


def kerasNeuralNetwork(imputar_datos, normalizar_datos, scaler_type, rebalancear, oversampling, undersampling,
                       smote_sampling, under_sampling_ratio, features, neuronas, epocas, batch, do_eval):
    # Leemos los archivos
    print("\nLeyendo archivos...")
    train = pd.read_csv("features_2/training.csv")
    validation = pd.read_csv("features_2/validation.csv")
    test = pd.read_csv("features_2/test.csv")
    random_state = np.random.randint(1000)

    X_train = train.loc[:, features]
    y_train = train.MSG_CLICKED
    X_validation = validation.loc[:, features]
    y_validation = validation.MSG_CLICKED
    X_test = test.loc[:, features]
    y_test = test.MSG_CLICKED

    if imputar_datos:
        print("Imputando datos...")
        # Por simpleza solo vamos a llenar los datos con el valor promedio
        # probablemente podemos hacer algo mejor...
        imputer = SimpleImputer().fit(X_train)
        X_train = imputer.transform(X_train)
        X_validation = imputer.transform(X_validation)
        X_test = imputer.transform(X_test)

    if normalizar_datos:
        print("Estandarizando datos...")
        # Vamos a utilizar RobustScaler que utiliza la mediana en lugar de la media
        # Es posible probar ambos y ver cual funciona mejor, dado que tenemos
        # datos con valores anomalos, considere que este era mejor.
        if scaler_type == "Standard":
            scaler = StandardScaler().fit(X_train)
        else:
            scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)

    if rebalancear:
        print("\nRebalanceando")
        if oversampling:
            sm = SMOTE(random_state=random_state, sampling_strategy=smote_sampling)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        if undersampling:
            under = RandomUnderSampler(random_state=random_state, sampling_strategy=under_sampling_ratio)
            X_train, y_train = under.fit_resample(X_train, y_train)

    # build a model
    model = Sequential()
    model.add(Dense(neuronas, input_shape=(X_train.shape[1],), activation='sigmoid'))  # Add an input shape! (features,)
    # model.add(Dense(neuronas, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Model metrics
    # compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['Accuracy', 'Precision', 'Recall'])
    # Training model
    model.fit(X_train, y_train, epochs=epocas, batch_size=batch)

    metrics = model.evaluate(X_validation, y_validation)
    print('VALIDATION METRICS: ', metrics)

    if do_eval:
        test_metrics = model.evaluate(X_test, y_test)
        print('TEST METRICS: ', test_metrics)

    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test, batch_size=batch, verbose=1)
    # train_metrics = precision_recall_fscore_support(y_test, y_pred, average='binary', labels=[0, 1])
    # print('\nTraining metrics')
    # print('Precision: ' + str(train_metrics[0]))
    # print('Recall: ' + str(train_metrics[1]))
    # print('F1: ' + str(train_metrics[2]))
    # print('MEAN Y: ', y_pred.mean())
    y_pred_bool = np.argmax(y_pred, axis=1)

    return


# print(classification_report(y_test, y_pred_bool))


def logisticRegression(imputar_datos, normalizar_datos, scaler_type,
                       add_categorical, cross_validation, rebalancear,
                       oversampling, undersampling, do_eval,
                       one_hot_frecuency, smote_sampling, under_sampling, features):
    precision = 0
    recall = 0
    f1 = 0
    random_state = np.random.randint(1000)

    # Leemos los archivos
    # print("\nLeyendo archivos...")
    train = pd.read_csv("features_2/training.csv")
    validation = pd.read_csv("features_2/validation.csv")
    test = pd.read_csv("features_2/test.csv")

    # FEATURE ENGINEERING
    # Seleccionamos las features que nos interesan
    feature_cols = features

    X = train.loc[:, feature_cols]
    y = train.MSG_CLICKED

    if imputar_datos:
        # print("Imputando datos...")
        # Por simpleza solo vamos a llenar los datos con el valor promedio
        # probablemente podemos hacer algo mejor...
        imputer = SimpleImputer()
        X = imputer.fit_transform(X)

    if normalizar_datos:
        # print("Estandarizando datos...")
        # Vamos a utilizar RobustScaler que utiliza la mediana en lugar de la media
        # Es posible probar ambos y ver cual funciona mejor, dado que tenemos
        # datos con valores anomalos, considere que este era mejor.
        if scaler_type == "Standard":
            scaler = StandardScaler().fit(X)
        else:
            scaler = RobustScaler().fit(X)
        X = scaler.transform(X)

    if add_categorical:
        # print("Creando features categoricas...")
        # Vamos a utilizar OneHotEncoding para usar las extensiones de los archivos
        # Tambien nos vamos a quedar solo con los valores mas comunes, esto requiere
        # busqueda de hiperparametros
        extensions = train.loc[:, ['STATE']].select_dtypes(include=[object])
        enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=one_hot_frecuency)
        enc.fit(extensions)
        extensionFeatures = enc.transform(extensions).toarray()
        X = pd.concat([pd.DataFrame(X), pd.DataFrame(extensionFeatures)], axis=1)

    # print("\nEvaluacion sin rebalanceo")
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    y_pred = clf.predict(X)
    train_metrics = precision_recall_fscore_support(y, clf.predict(X), average='binary', labels=[0, 1])
    # print('\nTraining metrics')
    # print('Precision: ' + str(train_metrics[0]))
    # print('Recall: ' + str(train_metrics[1]))
    # print('F1: ' + str(train_metrics[2]))

    X_val = validation.loc[:, feature_cols]
    y_val = validation.MSG_CLICKED
    if imputar_datos:
        X_val = imputer.transform(X_val)
    if normalizar_datos:
        X_val = scaler.transform(X_val)
    if add_categorical:
        val_extensions = validation.loc[:, ['STATE']].select_dtypes(include=[object])
        val_extensionFeatures = enc.transform(val_extensions).toarray()
        X_val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(val_extensionFeatures)], axis=1)
    val_metrics = precision_recall_fscore_support(y_val, clf.predict(X_val), average='binary', labels=[0, 1])
    # print('\nValidation metrics')
    # print('Precision: ' + str(val_metrics[0]))
    # print('Recall: ' + str(val_metrics[1]))
    # print('F1: ' + str(val_metrics[2]))
    # return val_metrics[0], val_metrics[1], val_metrics[2]

    if cross_validation:
        # print("\nEvaluando modelo SIN rebalanceo, utilizando cross fold validation")
        clf = LogisticRegression(random_state=random_state)
        cv = RepeatedStratifiedKFold(n_splits=10, random_state=random_state)
        scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
        # print("avg. roc auc: " + str(np.mean(scores)))

    if rebalancear:
        # print("\nRebalanceando")
        if oversampling:
            sm = SMOTE(random_state=random_state, sampling_strategy=smote_sampling)
            X, y = sm.fit_resample(X, y)
        if undersampling:
            under = RandomUnderSampler(random_state=random_state, sampling_strategy=under_sampling)
            X, y = under.fit_resample(X, y)

        # print("Evaluacion CON rebalanceo")
        clf = LogisticRegression(random_state=random_state).fit(X, y)
        train_metrics = precision_recall_fscore_support(y, clf.predict(X), average='binary', labels=[0, 1])
        # print('\nTraining metrics')
        # print('Precision: ' + str(train_metrics[0]))
        # print('Recall: ' + str(train_metrics[1]))
        # print('F1: ' + str(train_metrics[2]))

        X_val = validation.loc[:, feature_cols]
        y_val = validation.MSG_CLICKED
        if imputar_datos:
            X_val = imputer.transform(X_val)
        if normalizar_datos:
            X_val = scaler.transform(X_val)
        if add_categorical:
            val_extensions = validation.loc[:, ['STATE']].select_dtypes(include=[object])
            val_extensionFeatures = enc.transform(val_extensions).toarray()
            X_val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(val_extensionFeatures)], axis=1)
        val_metrics = precision_recall_fscore_support(y_val, clf.predict(X_val), average='binary',
                                                      labels=[0, 1])
        # print('\nValidation metrics')
        # print('Precision: ' + str(val_metrics[0]))
        # print('Recall: ' + str(val_metrics[1]))
        # print('F1: ' + str(val_metrics[2]))
        return val_metrics[0], val_metrics[1], val_metrics[2]

        if cross_validation:
            # print("\nEvaluando modelo CON rebalanceo, utilizando cross fold validation")
            clf = LogisticRegression(random_state=random_state)
            cv = RepeatedStratifiedKFold(n_splits=10, random_state=random_state)
            scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
            # print("avg. roc auc: " + str(np.mean(scores)))

    if do_eval:
        X = train.loc[:, feature_cols]
        y = train.MSG_CLICKED
        X_val = validation.loc[:, feature_cols]
        y_val = validation.MSG_CLICKED
        X_allTrain = pd.concat([pd.DataFrame(X), pd.DataFrame(X_val)])
        y_allTrain = np.concatenate((y, y_val), axis=0)
        X_test = test.loc[:, feature_cols]
        y_test = test.MSG_CLICKED
        if imputar_datos:
            imputer = SimpleImputer()
            X_allTrain = imputer.fit_transform(X_allTrain)
        if normalizar_datos:
            if scaler_type == "Standard":
                scaler = StandardScaler().fit(X_allTrain)
            else:
                scaler = RobustScaler().fit(X_allTrain)
            X_allTrain = scaler.transform(X_allTrain)
        if add_categorical:
            all_train = pd.concat([train, validation])
            extensions = all_train.loc[:, ['STATE']].select_dtypes(include=[object])
            enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=one_hot_frecuency)
            enc.fit(extensions)
            extensionFeatures = enc.transform(extensions).toarray()
            X_allTrain = pd.concat([pd.DataFrame(X_allTrain), pd.DataFrame(extensionFeatures)], axis=1)
        if rebalancear:
            if oversampling:
                sm = SMOTE(random_state=random_state, sampling_strategy=smote_sampling)
                X_allTrain, y_allTrain = sm.fit_resample(X_allTrain, y_allTrain)
            if undersampling:
                under = RandomUnderSampler(random_state=random_state, sampling_strategy=under_sampling)
                X_allTrain, y_allTrain = under.fit_resample(X_allTrain, y_allTrain)
        clf = LogisticRegression(random_state=random_state).fit(X_allTrain, y_allTrain)
        if imputar_datos:
            X_test = imputer.transform(X_test)
        if normalizar_datos:
            X_test = scaler.transform(X_test)
        if add_categorical:
            test_extensions = test.loc[:, ['STATE']].select_dtypes(include=[object])
            test_extensionFeatures = enc.transform(test_extensions).toarray()
            X_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(test_extensionFeatures)], axis=1)
        test_metrics = precision_recall_fscore_support(y_test, clf.predict(X_test), average='binary',
                                                       labels=[0, 1])
        # print('\nTesting metrics')
        # print('Precision: ' + str(test_metrics[0]))
        # print('Recall: ' + str(test_metrics[1]))
        # print('F1: ' + str(test_metrics[2]))
        return test_metrics[0], test_metrics[1], test_metrics[2]

    return precision, recall, f1


def logisticIteration(features, crossfold):
    logisticDF = pd.DataFrame()
    logisticDF['Model #'] = np.nan
    logisticDF['Over Ratio'] = np.nan
    logisticDF['Under Ratio'] = np.nan
    logisticDF['Precision'] = np.nan
    logisticDF['Recall'] = np.nan
    logisticDF['F1'] = np.nan

    # Iterating over and under sampling with features
    count = 0
    for i in np.arange(0.2, 0.51, 0.05):
        for j in np.arange(0.5, 0.81, 0.05):
            count = count + 1
            over_ratio = i
            under_ratio = j
            print("Model #", count, " over: ", over_ratio, " under: ", under_ratio, " features: ", len(features))
            precision_i, recall_i, f1_i = logisticRegression(imputar_datos=True, normalizar_datos=True,
                                                             scaler_type="Standard",
                                                             add_categorical=True, cross_validation=crossfold,
                                                             rebalancear=True,
                                                             oversampling=True, undersampling=True, do_eval=False,
                                                             one_hot_frecuency=1,
                                                             smote_sampling=over_ratio, under_sampling=under_ratio,
                                                             features=features)
            new_row = {'Model #': count, 'Over Ratio': over_ratio, 'Under Ratio': under_ratio,
                       'Precision': precision_i, 'Recall': recall_i, 'F1': f1_i}
            # logisticDF = logisticDF.append(new_row, ignore_index=True)
            logisticDF = pd.concat([logisticDF, pd.DataFrame.from_records([new_row])])
    return logisticDF


if __name__ == '__main__':

    logistic = True
    keras = True
    tree = True

    features1 = ['AGE_IN_NETWORK', 'VASTRIX_RECARGAS', 'BANKING', 'DEVICE_CLASS', 'USER_AGE', 'DATA_USER', 'TIKTOK',
                 'FACEBOOK', 'WHATSAPP', 'INSTAGRAM', 'SPOTIFY', 'AVG_NAV_AMNT_10', 'AVG_NAV_AMNT_20',
                 'AVG_NAV_AMNT_20PLUS',
                 'FIRST_MSG_SENT', 'SECOND_MSG_SENT']
    features2 = ['P_FREQ_H', 'P_FREQ_M', 'P_FREQ_L', 'P_COMPANY_TIGO',
                 'P_COMPANY_CLARO',
                 'P_COMPANY_INTL', 'P_PLAN_NOT_PREPAID', 'P_DEVICE2_SMARTPHONE', 'P_ARPU_GTE_100', 'P_ARPU_GTE_200',
                 'P_DAYS_DATA_USER_GTE_25', 'P_STATE_EQUALS', 'P_BANKING']
    features3 = ['AGE_IN_NETWORK', 'VASTRIX_RECARGAS', 'BANKING', 'DEVICE_CLASS', 'USER_AGE', 'DATA_USER', 'TIKTOK',
                 'FACEBOOK', 'WHATSAPP', 'INSTAGRAM', 'SPOTIFY', 'AVG_NAV_AMNT_10', 'AVG_NAV_AMNT_20',
                 'AVG_NAV_AMNT_20PLUS',
                 'FIRST_MSG_SENT', 'SECOND_MSG_SENT', 'P_FREQ_H', 'P_FREQ_M', 'P_FREQ_L', 'P_COMPANY_TIGO',
                 'P_COMPANY_CLARO',
                 'P_COMPANY_INTL', 'P_PLAN_NOT_PREPAID', 'P_DEVICE2_SMARTPHONE', 'P_ARPU_GTE_100', 'P_ARPU_GTE_200',
                 'P_DAYS_DATA_USER_GTE_25', 'P_STATE_EQUALS', 'P_BANKING']

    # Logistic regression
    if logistic:
        # resultFeatures1WOCF = logisticIteration(features1, False)
        #
        # resultFeatures2WOCF = logisticIteration(features2, False)
        #
        # resultFeatures3WOCF = logisticIteration(features3, False)
        #
        # print("#############  LOGISTIC FEATURES 1 WITH OUT CROSSFOLD #############")
        # print(resultFeatures1WOCF)
        # resultFeatures1WOCF.to_csv("logistic_results/logisticRegressionResultFeatures1WOCF.csv")
        # print("#############  LOGISTIC FEATURES 2 WITH OUT CROSSFOLD #############")
        # print(resultFeatures2WOCF)
        # resultFeatures2WOCF.to_csv("logistic_results/logisticRegressionResultFeatures2WOCF.csv")
        # print("#############  LOGISTIC FEATURES 3 WITH OUT CROSSFOLD #############")
        # print(resultFeatures3WOCF)
        # resultFeatures3WOCF.to_csv("logistic_results/logisticRegressionResultFeatures3WOCF.csv")

        precision_features1WCF, recall_features1WCF, f1_features1WCF = logisticRegression(imputar_datos=True,
                                                                                          normalizar_datos=True,
                                                                                          scaler_type="Robust",
                                                                                          add_categorical=True,
                                                                                          cross_validation=True,
                                                                                          rebalancear=True,
                                                                                          oversampling=True,
                                                                                          undersampling=True,
                                                                                          do_eval=True,
                                                                                          one_hot_frecuency=1,
                                                                                          smote_sampling=0.35,
                                                                                          under_sampling=0.8,
                                                                                          features=features1)

        precision_features2WCF, recall_features2WCF, f1_features2WCF = logisticRegression(imputar_datos=True,
                                                                                          normalizar_datos=True,
                                                                                          scaler_type="Robust",
                                                                                          add_categorical=True,
                                                                                          cross_validation=True,
                                                                                          rebalancear=True,
                                                                                          oversampling=True,
                                                                                          undersampling=True,
                                                                                          do_eval=True,
                                                                                          one_hot_frecuency=1,
                                                                                          smote_sampling=0.2,
                                                                                          under_sampling=0.8,
                                                                                          features=features2)

        precision_features3WCF, recall_features3WCF, f1_features3WCF = logisticRegression(imputar_datos=True,
                                                                                          normalizar_datos=True,
                                                                                          scaler_type="Robust",
                                                                                          add_categorical=True,
                                                                                          cross_validation=True,
                                                                                          rebalancear=True,
                                                                                          oversampling=True,
                                                                                          undersampling=True,
                                                                                          do_eval=True,
                                                                                          one_hot_frecuency=1,
                                                                                          smote_sampling=0.2,
                                                                                          under_sampling=0.8,
                                                                                          features=features3)

        print("#############  LOGISTIC FEATURES 1 WITH CROSSFOLD #############")
        print("Precision: ", precision_features1WCF, " - Recall: ", recall_features1WCF, " - F1: ",  f1_features1WCF)
        print("#############  LOGISTIC FEATURES 2 WITH CROSSFOLD #############")
        print("Precision: ", precision_features2WCF, " - Recall: ", recall_features2WCF, " - F1: ",  f1_features2WCF)
        print("#############  LOGISTIC FEATURES 3 WITH CROSSFOLD #############")
        print("Precision: ", precision_features3WCF, " - Recall: ", recall_features3WCF, " - F1: ",  f1_features3WCF)

    # Neural Network
    if keras:
        print("#############  NEURAL NETWORK FEATURES 1 #############")
        kerasNeuralNetwork(imputar_datos=True, normalizar_datos=True, scaler_type="Standard",
                           rebalancear=True, oversampling=True, undersampling=True,
                           smote_sampling=0.35, under_sampling_ratio=0.8, features=features1,
                           neuronas=32, epocas=10, batch=512, do_eval=True)

        print("#############  NEURAL NETWORK FEATURES 2 #############")
        kerasNeuralNetwork(imputar_datos=True, normalizar_datos=True, scaler_type="Standard",
                           rebalancear=True, oversampling=True, undersampling=True,
                           smote_sampling=0.2, under_sampling_ratio=0.8, features=features2,
                           neuronas=32, epocas=10, batch=512, do_eval=True)

        print("#############  NEURAL NETWORK FEATURES 3 #############")
        kerasNeuralNetwork(imputar_datos=True, normalizar_datos=True, scaler_type="Standard",
                           rebalancear=True, oversampling=True, undersampling=True,
                           smote_sampling=0.2, under_sampling_ratio=0.8, features=features3,
                           neuronas=32, epocas=10, batch=512, do_eval=True)

    # Decision Tree
    if tree:
        print("#############  DECISION TREE FEATURES 1 #############")
        decisionTree(imputar_datos=True, normalizar_datos=False, scaler_type="Standard",
                     rebalancear=True, oversampling=True, undersampling=True,
                     smote_sampling=0.35, under_sampling_ratio=0.8, features=features1)

        print("#############  DECISION TREE FEATURES 2 #############")
        decisionTree(imputar_datos=True, normalizar_datos=False, scaler_type="Standard",
                     rebalancear=True, oversampling=True, undersampling=True,
                     smote_sampling=0.2, under_sampling_ratio=0.8, features=features2)

        print("#############  DECISION TREE FEATURES 3 #############")
        decisionTree(imputar_datos=True, normalizar_datos=False, scaler_type="Standard",
                     rebalancear=True, oversampling=True, undersampling=True,
                     smote_sampling=0.2, under_sampling_ratio=0.8, features=features3)
