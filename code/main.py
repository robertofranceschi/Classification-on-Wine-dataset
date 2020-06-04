# References // Useful links
# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html
# https://gist.github.com/anandology/772d44d291a9daa198d4
# Data standardization/normalization
# https://stats.stackexchange.com/questions/287425/why-do-you-need-to-scale-data-in-knn
# https://stats.stackexchange.com/questions/363889/which-type-of-data-normalizing-should-be-used-with-knn


### Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from utils import *
#from models import * 
import sys


### Custom parameters
STRATIFIED = True           # split stratified (i.e. maintain the portion of classes in each subset)
NORMALIZE_DATA = True       # normalize data 
SHOW_IMAGES = False         # show images on output (if False the program will just save the images in the folder)
SHOW_INFO_DATASET = False   # print dataset description, sets portion and other info

# Classifier
KNN = True             # K-Nearest Neighbors with K = [1,3, 5,7]
LINEAR_SVM = True      # linear SVM with C = [0.001, 0.01, 0.1, 1, 10, 100,1000]
KERNEL_SVM = True      # RBF kernel with C = [0.001, 0.01, 0.1, 1, 10, 100,1000]
GRID_SEARCH = True     # grid search for an RBF kernel {gamma,C}
CROSS_VALID = True     # grid search for gamma and C but this time perform 5-fold validation

# Extra
DO_PCA = True               # Principal Component Analysis
SELECT_FEATURES = True      # Feature selection
RUN_SELECT_FEATURES = False # If 'True' runs the classifications with the best 2 features: 'color_intensity', 'proline' (previously founded with SelectKBest feature)

# -------------------

# Load Wine dataset (from scikit library) 
wine = load_wine()

if SHOW_INFO_DATASET : 
    # Dataset description 
    print(wine['DESCR'])

# 2. Select the first two attributes for a 2D representation of the image
X = wine.data[:,:2]
if RUN_SELECT_FEATURES : 
    X = wine.data[:,[9,12]] # Feature selection (best 2 features found)
y = wine.target

# Show 2D Representation
plot2Drepresentation(X,y,show=SHOW_IMAGES)

# Split data into train, validation and test sets in proportion 5:2:3 
if STRATIFIED : 
    X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size=0.3,stratify=y) # test = 0.3 - train = 0.7 #X_norm
    X_train,X_val,y_train,y_val = train_test_split(X_train_val,y_train_val,test_size=0.2/0.7,stratify=y_train_val) # train = 0.5 - val = 0.2 
else : 
    # without using stratified results strongly depend on random parameters
    X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size=0.3) # test = 0.3 - train = 0.7 #X_norm
    X_train,X_val,y_train,y_val = train_test_split(X_train_val,y_train_val,test_size=0.2/0.7) # train = 0.5 - val = 0.2 

if SHOW_INFO_DATASET: 
    # print set dimension
    print_sets_dimensions(X,X_train,X_val,X_test,y_train,y_val,y_test)
    # print element per class for each set -> if STARTIFIED=True the portion of each class must be the same 
    check_element_per_class(y_train,y_val,y_test)

if NORMALIZE_DATA : 
    # Normalize data
    scaler = StandardScaler()
    scaler.fit(X_train_val)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train_val = scaler.transform(X_train_val)

# Setup plot
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

# define colors
cmap_light = ListedColormap(['#75ff76', '#ff8cff', '#7a7aff'])
cmap_bold = ListedColormap(['#00ff01', '#ff00fe', '#0000fe'])

# K-Nearest Neighbors
if KNN : 
    print("\n---- KNN ----")
    k_values = [1,3,5,7]
    scores = []
    #plt.figure()
    ax = [None]*len(k_values)
    for i, k in enumerate(k_values):    
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"\t[k = {k}] \tAccuracy (validation): {acc:.5f}")
        scores.append(acc)

        # Create plot        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)        
            
        ax[i] = plt.subplot(2, 2, i + 1)    
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')
        plt.title(f"K = {k}", fontsize=9)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())    
    
    clear_labels(ax, [0, 1], [1, 3])
    plt.savefig('images/Figure2_knn_PredPlot', dpi=250)

    # Plot accuracy on validation set
    plot_accuracy_validation(k_values,scores,param_name='K',output="images/Figure3_knn_acc",show=SHOW_IMAGES)

    # best K value
    bestK = k_values[np.argmax(np.array(scores))]
    # evaluate the model on the test set with the best param
    best_knn = KNeighborsClassifier(n_neighbors=bestK, n_jobs=-1).fit(X_train_val, y_train_val)
    print(f"> [Best value of K: {bestK}]\n> Accuracy (test set): {best_knn.score(X_test, y_test):.3f}")


# Linear SVM
if LINEAR_SVM :
    print("\n---- LINEAR SVM ----") 
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #regularization parameter
    scores = []
    plt.figure()
    ax = [None]*len(C_values)
    for i, c in enumerate(C_values):    
        clf = LinearSVC(C=c,max_iter=1000000).fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"\t[C = {c}] \tAccuracy (validation): {acc:.5f}")
        scores.append(acc)

        ax[i] = plt.subplot(3, 3, i+1) 
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')
        plt.title(f"C = {c}", fontsize=9)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)

    clear_labels(ax, [0, 1, 2, 3], [1, 2, 4, 5])
    plt.savefig('images/Figure4_svm_PredPlot', dpi=250)

    # Plot accuracy on validation set
    plot_accuracy_validation(C_values,scores,log=True,param_name='C',output="images/Figure5_svm_acc",show=SHOW_IMAGES)

    # best C value
    bestC_svm = C_values[np.argmax(np.array(scores))]
    # evaluate the model on the test set with the best param
    clf = LinearSVC(C=bestC_svm,max_iter=1000000).fit(X_train_val, y_train_val)
    print(f"> [Best value of C: {bestC_svm}]\n> Accuracy (test set): {clf.score(X_test, y_test):.3f}")

# RBF kernel
if KERNEL_SVM :
    print("\n---- KERNEL SVM ----")
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #regularization parameter
    scores = []
    plt.figure()
    ax = [None]*len(C_values)
    for i, c in enumerate(C_values):    
        clf = SVC(C=c, kernel= 'rbf').fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"\t[C = {c}] \tAccuracy (validation): {acc:.5f}")
        scores.append(acc)

        ax[i] = plt.subplot(3, 3, i+1) 
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')
        plt.title(f"C = {c}", fontsize=9)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)

    clear_labels(ax, [0, 1, 2, 3], [1, 2, 4, 5])
    plt.savefig('images/Figure6_rbf_PredPlot', dpi=250)

    # Plot accuracy on validation set
    plot_accuracy_validation(C_values,scores,log=True,param_name='C',output="images/Figure7_rbf_acc",show=SHOW_IMAGES)

    # best C value
    bestC_rbf = C_values[np.argmax(np.array(scores))]
    # evaluate the model on the test set with the best param
    clf = LinearSVC(C=bestC_rbf,max_iter=1000000).fit(X_train_val, y_train_val)
    print(f"> [Best value of C: {bestC_rbf}]\n> Accuracy (test set): {clf.score(X_test, y_test):.3f}")


    if GRID_SEARCH :
        print("\n---- KERNEL SVM + GRID SEARCH (C,gamma) ----")

        gamma_values = [1e-5,1e-3,1e-2,1e-1,1,1e1,1e2,1e3]
        GridScore = np.zeros((len(gamma_values), len(C_values)))
        for i,c in enumerate(C_values) : 
            for j,gamma in enumerate(gamma_values) :
                clf = SVC(C=c, gamma=gamma, kernel='rbf').fit(X_train, y_train)
                GridScore[j][i] = clf.score(X_val, y_val)
                print(f"\t[C = {c}, gamma = {gamma}] \tAccuracy (validation): {GridScore[j][i]:.5f}")
        
        #find best parameters
        #print(GridScore)
        #print(np.argmax(GridScore))
        #best_rbf_grid_score = np.argmax(GridScore)
        bestC = C_values[int(np.argmax(GridScore)/7)]
        bestGamma = gamma_values[np.argmax(GridScore)%7]
        # plot results
        plt.figure()
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,edgecolor='k', s=20)      
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')    
        plt.savefig('./images/Figure8_gridPredPlot.png', dpi=250)

        # evaluate on test set with best parameters found
        best_rbf_grid_score = SVC(C=bestC, gamma=bestGamma, kernel='rbf').fit(X_train_val, y_train_val)
        print(f"> [Best params: C={bestC}, gamma={bestGamma}]")
        print(f"> Accuracy (test set): {best_rbf_grid_score.score(X_test, y_test):.3f}")

    # Grid search with 5-Fold CV
    if CROSS_VALID : 
        print("\n---- KERNEL SVM + CROSS VAL + GRID SEARCH ----")
        # Merge train and validation sets
        parameters = {'gamma': (1e-5,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,'auto', 'scale'), 'C': C_values}
        clf = GridSearchCV(SVC(kernel="rbf"), parameters, cv=5, iid='false').fit(X_train_val, y_train_val)
        plt.figure()
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train_val[:, 0], X_train_val[:, 1], c=y_train_val, cmap=cmap_bold,edgecolor='k', s=20)    
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')    
        plt.savefig('./images/Figure9_CV_GridPredPlot.png', dpi=250)
        gridTestScore = clf.score(X_test, y_test)
        bestParams = ', '.join("{!s}={!r}".format(key,val) for (key,val) in clf.best_params_.items())    
        print(f"> [Best params: {bestParams}]")
        print(f"> Accuracy (test set): {gridTestScore:.3f}")

# Try also with different pairs of attributes
# --- Principal Component Analysis
if DO_PCA :
    print("\n---- PCA ----")
    pca = PCA(n_components=2)
    pca.fit(X_train_val)
    pca_train = pca.transform(X_train)
    pca_test = pca.transform(X_test)

    if KNN : 
        clf = KNeighborsClassifier(n_neighbors=bestK).fit(pca_train, y_train)      
        score = clf.score(pca_test,y_test)
        print(f"KNN score: {score:.3f}")

    if LINEAR_SVM : 
        clf = LinearSVC(C=bestC_svm, max_iter=1000000) .fit(pca_train, y_train)
        score = clf.score(pca_test,y_test)
        print(f"linear_SVM score: {score:.3f}")

    if KERNEL_SVM :
        #todo: probabilemente SVM
        clf = SVC(C=bestC_rbf, gamma='scale', kernel='rbf').fit(pca_train, y_train)
        score = clf.score(pca_test,y_test)
        print(f"kernel_SVM score: {score:.3f}") 

# Feature selection
if SELECT_FEATURES : 
    print("\n---- Feature selection ----")
    X_select = SelectKBest(chi2, k=2).fit_transform(wine.data, y)
    #print(wine.data.shape)
    #print(X_select.shape)
    #print(X_select[:5,:])
    #print(wine.data[:5])
    #print(wine.feature_names)
    # best features found
    print("best feature found: {'color_intensity' : 9, 'proline', 12} using chi2")
    print("best feature found: {'flavanoids' : 7, 'proline', 12} using f_classif")


