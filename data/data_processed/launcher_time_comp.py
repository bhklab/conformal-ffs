

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import pickle
import os


from sklearn.svm import  LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV


from sklearn.svm import  LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import   classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, has_fit_parameter

#from crepes.extras import hinge

from CRFE_toolbox.IA_conformal import training, calibration, run,  Conformal_prediction
#from CRFE_toolbox.Plotter import  plot_classical_performance , plot_features_behaviour , check_consistency 

#sys.path.insert(0, "../") 
from main import FeatureSelector
from Utils import USER, classical_report, hinge
from Plotter import plotter


num_cpus = os.cpu_count()
print("Number of CPUs available:", num_cpus)



base_seed  = 2
np.random.seed(base_seed)

_method = "mRMR"
print("METHOD: ", _method)
dataset_user = "semeion"
y_classes = [0,1,2,3,4,5,6,7,8,9]

lambda_ = 0.5
confidence_level  = 0.1

Lambda_p = (1-lambda_) / (len(y_classes)-1)

 
# IMV_ABA
#train_df = pd.read_csv('../DATASETS/'+dataset_user+'/train_IMVIGOR.csv')
#test_df = pd.read_csv('../DATASETS/'+dataset_user+'/test_ABACUS.csv')
#df = pd.concat([train_df, train_df], ignore_index=True)
#Y = df['therapy_effect']
#df = df.drop(columns=[ "DFS_months", "ID", "estudio", 'therapy_effect'])



df = pd.read_csv('../DATASETS/'+ dataset_user+ '/data_'+ dataset_user + '_standarized.csv')


#n_samples = 1000
#n_features = 100  # Total number of features
#n_informative = 10  # Number of informative features
#n_classes = 3  # Number of classes
#X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
#                           n_classes=n_classes, random_state=123456, weights= (0.3,0.31, (1.-0.3-0.31)), flip_y = 0.1 )
#df = pd.DataFrame(X)
#df['Class'] = y


Y = df['Class']
df = df.drop(columns=['Class'])

# Convert dataframes to numpy arrays
X = df.to_numpy()
Y = Y.to_numpy()

n_features = X.shape[1]

#X = X[:, :100]
#X_test = X_test[:, :1000]


list_of_dicc_SVM = []
list_of_class_dicc_SVM = []

list_of_dicc_GB = []
list_of_class_dicc_GB = []

jj = 0
ii = 0

kf = KFold(n_splits = 5, shuffle=True, random_state = base_seed + ii )
for train_index, test_index in kf.split(X):

    X_tr, X_test = X[train_index], X[test_index]
    Y_tr, Y_test = Y[train_index], Y[test_index]


    OUT = {}
    list_of_names = ["Index", "coverage", "inefficiency", 
                    "certainty", "uncertainty", "mistrust",
                    "S_score", "F_score", "Creditibily"]
        
    for i in range(0, len( list_of_names)):
        OUT[list_of_names[i]] = []

        
    OUT_CLASSIC = {}
    list_of_classical_names = ['accuracy', 'precision', 'recall', 
                              'f1_micro', 'f1_macro', 'f1_weighted','per_class']
        
    for i in range(0, len( list_of_classical_names)):
        OUT_CLASSIC[list_of_classical_names[i]] = []
        



    #from sklearn.svm import  LinearSVC

    estimator = LinearSVC(tol = 1.e-4, 
                          loss='squared_hinge',
                          max_iter= 200000,
                          dual = True,
                          class_weight = "balanced",
                          random_state= base_seed + ii )
    
    
    sele = FeatureSelector(estimator = estimator,
                           classes_ = y_classes,
                           max_features = -1,
                           out_cp = OUT,
                           out_class = OUT_CLASSIC
                             )
    

    OUT_SVM = {}
    list_of_names = ["Index", "coverage", "inefficiency", 
                    "certainty", "uncertainty", "mistrust",
                    "S_score", "F_score", "Creditibily"]
        
    for i in range(0, len( list_of_names)):
        OUT_SVM[list_of_names[i]] = []

        
    OUT_CLASSIC_SVM = {}
    list_of_classical_names = ['accuracy', 'precision', 'recall', 
                              'f1_micro', 'f1_macro', 'f1_weighted','per_class']
        
    for i in range(0, len( list_of_classical_names)):
        OUT_CLASSIC_SVM[list_of_classical_names[i]] = []



    OUT_GB = {}
    list_of_names = ["Index", "coverage", "inefficiency", 
                    "certainty", "uncertainty", "mistrust",
                    "S_score", "F_score", "Creditibily"]
        
    for i in range(0, len( list_of_names)):
        OUT_GB[list_of_names[i]] = []

        
    OUT_CLASSIC_GB = {}
    list_of_classical_names = ['accuracy', 'precision', 'recall', 
                              'f1_micro', 'f1_macro', 'f1_weighted','per_class']
        
    for i in range(0, len( list_of_classical_names)):
        OUT_CLASSIC_GB[list_of_classical_names[i]] = []
    
    #OUT_SVM = OUT.copy()
    #OUT_GB = OUT.copy()
    #OUT_CLASSIC_SVM = OUT_CLASSIC.copy()
    #OUT_CLASSIC_GB = OUT_CLASSIC.copy()
    
    if _method == "mRMR_MS":
        salida = sele.mRMR_MS(X_tr, Y_tr,  split_size = 0.5)

    elif _method == "mRMR":
        salida = sele.mRMR(X_tr, Y_tr)

    else:
        exit()

    print(salida)


    #-----------------------------------------

    # CLASICAL PREDICTION

    #----------------------------------------
    j = 0
    for selected_features in salida:

        print("predicting", j, "/", len(salida) )

        X_tr_new = X_tr[:, selected_features].copy()   
        X_test_new = X_test[:, selected_features].copy() 

        print(X_tr_new.shape, Y_tr.shape)

        X_tr_new , X_cal_new , Y_tr_new, Y_cal = train_test_split( X_tr_new, Y_tr, test_size=0.45,  shuffle = True, stratify=Y_tr, 
                                                                   random_state= base_seed + ii)



        estimator = LinearSVC(tol = 1.e-4, 
                              loss='squared_hinge', 
                              max_iter= 300000,
                              multi_class = "ovr",
                              class_weight = "balanced",
                              random_state=  base_seed + ii )


        estimator.fit(X_tr_new , Y_tr_new)

        out_classic = []
        y_pred = estimator.predict(X_test_new)
            
        out_classic.append(accuracy_score(Y_test, y_pred))

        out_classic.append(precision_score(Y_test, y_pred, average = 'macro'))
        out_classic.append( recall_score( Y_test, y_pred, average = 'macro'))
        out_classic.append(f1_score( Y_test, y_pred, average = 'micro'))
        out_classic.append(f1_score( Y_test, y_pred, average = 'macro'))
        out_classic.append(f1_score(Y_test, y_pred,  average = 'weighted'))

        cl_report = classification_report(Y_test, y_pred , output_dict = True)
        out_classic.append(classical_report(cl_report)) # per_class metrics
        print("SVM CLASSIC: ",out_classic)
        for i in range(0, len( list_of_classical_names)):
            OUT_CLASSIC_SVM[list_of_classical_names[i]] += [out_classic[i]]
    

    #-----------------------------------------

    # CONFORMAL PREDICTION

    #----------------------------------------
    

        w_2, bias = training().SVMl(X_tr_new, Y_tr_new)
        
        NCM_cal,qhat = calibration(confidence_level).linear_distance(X_cal_new, Y_cal, w_2, 
                                                                        bias, len(y_classes),
                                                                        lambda_, Lambda_p)

        NCM = run().linear_distance(X_test_new, y_classes, w_2, 
                                    bias, len(y_classes),
                                    lambda_, Lambda_p) # NCM of tests saples
           

        
        output_path = "prueba"
        out = Conformal_prediction(NCM, NCM_cal, confidence_level, y_classes, Y_test, output_path, Flag = True) #predition sets

        
    
        OUT_SVM["Index"] += [selected_features]
        for i in range(1, len( list_of_names)):
            OUT_SVM[list_of_names[i]] += [out[i-1]] # ya que out no incuye index debe empezar en 0, no en 1
    
        #print("selected features: ", selected_features  )



        ############################
        #                          #
        ## GRADIENT BOOSTING PART ##
        #                          #
        ############################



        xgb = GradientBoostingClassifier(n_estimators=500, 
                                        learning_rate=.1 ,
                                        max_depth =2, 
                                        random_state= base_seed + ii)
        
        xgb.fit(X_tr_new , Y_tr_new)

        out_classic = []
        y_pred = xgb.predict(X_test_new)
            
        out_classic.append(accuracy_score(Y_test, y_pred))

        out_classic.append(precision_score(Y_test, y_pred, average = 'macro'))
        out_classic.append( recall_score( Y_test, y_pred, average = 'macro'))
        out_classic.append(f1_score( Y_test, y_pred, average = 'micro'))
        out_classic.append(f1_score( Y_test, y_pred, average = 'macro'))
        out_classic.append(f1_score(Y_test, y_pred,  average = 'weighted'))

        cl_report = classification_report(Y_test, y_pred , output_dict = True)
        out_classic.append(classical_report(cl_report)) # per_class metrics
        print("GB CLASSIC: ", out_classic)
        for i in range(0, len( list_of_classical_names)):
            OUT_CLASSIC_GB[list_of_classical_names[i]] += [out_classic[i]]
    

    #-----------------------------------------

    # CONFORMAL PREDICTION

    #----------------------------------------
    

        xgb.fit(X_tr_new, Y_tr_new)
        NCM_cal = hinge(xgb.predict_proba(X_cal_new), y_classes, Y_cal)

        NCM = hinge(xgb.predict_proba(X_test_new)) 
              
        output_path = "prueba"
        out = Conformal_prediction(NCM, NCM_cal, confidence_level, y_classes, Y_test, output_path, Flag = True) #predition sets


        OUT_GB["Index"] += [selected_features]
        for i in range(1, len( list_of_names)):
            OUT_GB[list_of_names[i]] += [out[i-1]] # ya que out no incuye index debe empezar en 0, no en 1




        j = j + 1


        

    list_of_dicc_SVM.append(OUT_SVM)
    list_of_class_dicc_SVM.append(OUT_CLASSIC_SVM)

    list_of_dicc_GB.append(OUT_GB)
    list_of_class_dicc_GB.append(OUT_CLASSIC_GB)


    ii = ii + 1 # update the seed



with open(os.path.join(r"save/save_" + dataset_user , dataset_user  +  "_" + _method  +  "_conformal_" + "SVM"  + '.pickle'), 'wb') as f:
    pickle.dump(list_of_dicc_SVM, f)

with open(os.path.join(r"save/save_" + dataset_user , dataset_user  +  "_" + _method + "_classic_"  + "SVM" +   '.pickle'), 'wb') as f:
    pickle.dump(list_of_class_dicc_SVM, f)



with open(os.path.join(r"save/save_" + dataset_user , dataset_user  +  "_" + _method  +  "_conformal_" + "XGB"  + '.pickle'), 'wb') as f:
    pickle.dump(list_of_dicc_GB, f)

with open(os.path.join(r"save/save_" + dataset_user , dataset_user  +  "_" + _method + "_classic_"  + "XGB" +   '.pickle'), 'wb') as f:
    pickle.dump(list_of_class_dicc_GB, f)



plotter().plot_conformal(dataset_user,"SVM", len(y_classes) )
plotter().plot_conformal(dataset_user,"XGB", len(y_classes) )

plotter().plot_stability_comp( dataset_user, n_features, alpha = 0.1)