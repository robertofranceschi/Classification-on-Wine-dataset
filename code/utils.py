import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

def check_element_per_class(y_train,y_val,y_test) : 
    # print element per class for each set
    print("Train: ", end='')
    print(Counter(y_train))
    print("Val: ", end='')
    print(Counter(y_val))
    print("Test: ", end='')
    print(Counter(y_test))

    return 

def print_sets_dimensions(X,X_train,X_val,X_test,y_train,y_val,y_test) : 
    # print dimension of sets
    print(f"X_train:\t{X_train.shape}\t\ty_train:\t{y_train.shape}\t{X_train.shape[0]/X.shape[0]}")     #0.4943820224719101 (88,)
    print(f"X_val:\t\t{X_val.shape}\t\ty_val:\t\t{y_val.shape}\t{X_val.shape[0]/X.shape[0]}")               #0.2022471910112359 (36,)
    print(f"X_test:\t\t{X_test.shape}\t\ty_test:\t\t{y_test.shape}\t{X_test.shape[0]/X.shape[0]}")          #0.3033707865168539 (54,)
    print()
    return 

def plot2Drepresentation(data,groups,show=False) : 
    cmap_bold = ['#00ff01', '#ff00fe', '#0000fe']
    c_groups = [cmap_bold[i] for i in groups]
    plt.scatter(data[:,0],data[:,1],c=c_groups,edgecolors='black')
    #plt.title('2D representation', )
    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    if show : 
        plt.show()
    plt.savefig('./images/Figure1_2D_representation.png', dpi=250)
    return

def plot_accuracy_validation(params,scores,param_name,log=False,output="void",show=False) : 
    plt.figure()
    plt.scatter(params,scores,c="#4287f5")
    if log == True : 
        plt.xlim(0.0001, 10000)
        plt.xscale("log") 
    plt.xlabel(param_name)
    plt.xticks(params)
    plt.ylabel('accuracy')
    if show : 
        plt.show()
    plt.savefig(output, dpi=250)

def clear_labels(ax, x, y):
  for i, c in enumerate(x):
    ax[c].set_xticklabels([])
    ax[c].set_xlabel('')    
  for i, c, in enumerate(y):
    ax[c].set_yticklabels([])
    ax[c].set_ylabel('') 
