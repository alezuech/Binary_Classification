import pandas as pd
import matplotlib.pyplot as plt
import os.path
import os

df = pd.read_csv('object.csv')

df.loc[df["object"] == "object1", 'object'] = 0
df.loc[df["object"] == "object2", 'object'] = 1

attr_list = ['attr1','attr2','attr3','attr4','attr5']

# column-wise normalization
df=(df-df.min())/(df.max()-df.min())
df['object'] = df['object'].astype('int')

attr_list2 = attr_list
if not os.path.isdir('plots'): os.mkdir('plots')

for at1 in attr_list:
    attr_list2.remove(at1)
    for at2 in attr_list2:
        size_=15
        alpha_=1
        
        object11 = df.loc[df['object'] == 0][at1].tolist()
        object12 = df.loc[df['object'] == 0][at2].tolist()
        
        object21 = df.loc[df['object'] == 1][at1].tolist()
        object22 = df.loc[df['object'] == 1][at2].tolist()
        
        plt.scatter(object11,object12)
        plt.scatter(object21,object22)
        
        plt.xlabel(at1)
        plt.ylabel(at2)
        plt.savefig(('plots/scatter_'+at1+'_'+at2+'.png').strip(), dpi=300, bbox_inches="tight")
        plt.close('all')


for at in attr_list:
    object1 = df.loc[df['object'] == 0][at]
    object2 = df.loc[df['object'] == 1][at]
    
    if at == 'attr5': bins = 30
    else: bins = 70
        
    plt.hist(object1, bins, alpha=0.5, label='object1')
    plt.hist(object2, bins, alpha=0.5, label='object2')
    plt.legend(loc = 'upper right')
    plt.title(at)
    plt.savefig(('plots/hist_'+ at+'.png').strip(), dpi=300)
    plt.close('all')