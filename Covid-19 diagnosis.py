
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row', 111)
#afficher toutes les colonnes du dataset:
pd.set_option('display.max_column', 111)

data=pd.read_excel('dataset.xlsx')
data.head()
df=data.copy()
df.shape
df.dtypes
df.dtypes.value_counts()
df.dtypes.value_counts().plot.pie()
df.isna()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

(df.isna().sum()/df.shape[0]).sort_values(ascending=True)

df=df[df.columns[df.isna().sum()/df.shape[0] <0.9]]
df.head()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)


df=df.drop('Patient ID', axis=1)

df['SARS-Cov-2 exam result'].value_counts()

df['SARS-Cov-2 exam result'].value_counts(normalize=True)


for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])


df['Patient age quantile'].value_counts()


df['SARS-Cov-2 exam result'].unique()


for col in df.select_dtypes('object'):
    print(f'{col:-<50} {df[col].unique()}')


for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()

positive_df=df[df['SARS-Cov-2 exam result']=='positive']


negative_df=df[df['SARS-Cov-2 exam result']=='negative']


missing_rate=df.isna().sum()/df.shape[0]

blood_columns=df.columns[(missing_rate<0.9) & (missing_rate>0.88)]
blood_columns

viral_columns=df.columns[(missing_rate<0.88) & (missing_rate>0.75)]

for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()

sns.countplot(x='Patient age quantile',hue='SARS-Cov-2 exam result',data=df)


for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')


sns.pairplot(df[blood_columns])

sns.heatmap(df[blood_columns].corr())

sns.clustermap(df[blood_columns].corr())

for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile',y=col,hue='SARS-Cov-2 exam result',data=df)

df.corr()['Patient age quantile'].sort_values()

pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])

df['est malade']= np.sum(df[viral_columns[:-2]]=='detected', axis=1) >=1

malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]

for col in blood_columns:
    plt.figure()
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()



def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'

df['statut'] = df.apply(hospitalisation, axis=1)

for col in blood_columns:
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col], label=cat)
    plt.legend()

df[blood_columns].count()

df[viral_columns].count()


df1 = df[viral_columns[:-2]]
df1['covid'] = df['SARS-Cov-2 exam result']
df1.dropna()['covid'].value_counts(normalize=True)


df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna()['covid'].value_counts(normalize=True)


from scipy.stats import ttest_ind

balanced_neg = negative_df.sample(positive_df.shape[0])

def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else :
        return 0

for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')

df=data.copy()
df.head()



missing_rate=df.isna().sum()/df.shape[0]

blood_columns=list(df.columns[(missing_rate<0.9) & (missing_rate>0.88)])
viral_columns=list(df.columns[(missing_rate<0.88) & (missing_rate>0.75)])

key_columns=['Patient age quantile','SARS-Cov-2 exam result']


df = df[key_columns + blood_columns + viral_columns]
df.head()

from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(df, test_size=0.2, random_state=0)


trainset['SARS-Cov-2 exam result'].value_counts()


testset['SARS-Cov-2 exam result'].value_counts()

code={'positive':1,
       'negative':0,
       'detected':1,
       'not_detected':0}

for col in df.select_dtypes('object'):
        df[col] = df[col].map(code)

df.dtypes.value_counts()


def encodage(df):
    code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}
    
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
    return df

def feature_engineering(df):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1 # suffit que la personne  aurait au moins une maladie
    df = df.drop(viral_columns, axis=1) #on elimine toute les variables du type viral a l'exeption du variable "est malade"
    return df


def imputation(df):
   # df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return  df

def preprocessing(df):
    
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    
    X = df.drop('SARS-Cov-2 exam result', axis=1) #on elemine la colonne sars....
    y = df['SARS-Cov-2 exam result']
    
    print(y.value_counts()) #on imprime un petit rapport nous indiquant la taille de y(le nombre de cas positif et le nombre de
                            # de cas negative de se trouvent ds y) apres avoir effectuer l'imputation 
    
    return X, y


# In[95]:


#on va faire passer notre trainset et testset dans notre fct:
X_train, y_train = preprocessing(trainset)

X_test, y_test = preprocessing(testset)



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))


RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))#on ajoute, entre preprocessor et svm, on ajoute une
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())


dict_of_models = {'RandomForest': RandomForest,
                  'AdaBoost' : AdaBoost,
                  'SVM': SVM,
                  'KNN': KNN
                 }

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

def evaluation(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    #print(classification_report(y_test, ypred))
    
    #ajout des learning_curve 
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    print(N,train_sc)
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()



for name, model in dict_of_models.items():
    print(name)
    evaluation(model)
    

#OPTIMIZATION:
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#lorque on utilise GridSearchCV il faut qu'on defenissent un dictionnaire  d'hyperparametres:
hyper_params = {'svc__gamma':[1e-3, 1e-4],
                'svc__C':[1, 10, 100, 1000], 
                'pipeline__polynomialfeatures__degree':[2, 3],
                'pipeline__selectkbest__k': range(40, 60)}


grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4,
                          n_iter=40)
#on entraine notre grid:
grid.fit(X_train, y_train)

#on va afficher les meilleur hyperparams:
print(grid.best_params_) #best_params_ est un attribut de la classe GridSearchCV.

#on peut calculer un vecteur prediction:
y_pred = grid.predict(X_test)

evaluation(grid.best_estimator_)
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()
def model_final(model, X, threshold=0):
    return model.decision_function(X) > threshold

y_pred = model_final(grid.best_estimator_, X_test, threshold=-1)

f1_score(y_test, y_pred)

from sklearn.metrics import recall_score
recall_score(y_test, y_pred)

pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12, 8))

