#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


#pour afficher toutes les donnees de notre dataset
pd.set_option('display.max_row', 111)
#afficher toutes les colonnes du dataset:
pd.set_option('display.max_column', 111)


# In[14]:


data=pd.read_excel('dataset.xlsx')


# In[15]:


data.head()


# In[16]:


df=data.copy()


# In[17]:


#identifier le nombre de lignes et de colonnes
df.shape


# In[18]:


#pour le type de variable(qualitative ou quantitative) on utilise la fonction dtypes de pandas
df.dtypes


# In[19]:


#pour compter le nombre de type de variable on utilise la fct value_counts
df.dtypes.value_counts()


# In[20]:


#pour transformer ces valeurs en graphe en utilise:
df.dtypes.value_counts().plot.pie()


# In[21]:


#analyse des valeurs manquantes de notre dataset:
#en prend notre dataset et en utilise la fct isna() de pandas et cette fct verifie si une valeur est na (not a number)
df.isna()


# In[22]:


#si on veut afficher cela avec seaborn et on voit toute notre dataset avec toute ces lignes et toutes ces colonnes et en blanc 
#on voit la ou il y a des valeurs manquantes
plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)


# In[23]:


#pour mésurer le pourcentage de valeurs manquantes qui nous manque dans nos differente colonnes, on fait la somme des valeurs
#manquantes qu'on trouve dans tous nos colonnes et pour faire le % on va diviser par le nombre de lignes de notre tableaux avec
#df.shape[0], et nous donne un %, puis on trie ces valeurs dans l'ordre croissant ou decroi avec la fct sort_values
#on remarque qu'on a des variables avec aucun valeur manquates, puis des variables avec 76% de valeurs manquates et des variables
#avec 89% des valeurs manquantes 
(df.isna().sum()/df.shape[0]).sort_values(ascending=True)


# In[24]:


#Analyse de fond
#1-Visualisation initiale-Elimination des colonnes inutiles: eliminer des colones qui ont 90% ou plus des valeurs manquantes
#pour eliminer avec une seule ligne de code toutes les variables ou il y a plus de 90% des NaN
df=df[df.columns[df.isna().sum()/df.shape[0] <0.9]]
df.head()
#on a passer de 111 colonnes a 39 colonnes


# In[25]:


plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)


# In[26]:


#on elimine la colonne Patient ID qui nous servira a rien: axis=1 c'est l'axe des colonne
df=df.drop('Patient ID', axis=1)


# In[27]:


df.head()


# In[17]:


#Visualisation de la target:
#Examen de la colonne target:


# In[28]:


#on va commencer par compter le nombre de cas positif et nbre de cas negatif avec value_counts:
df['SARS-Cov-2 exam result'].value_counts()


# In[29]:


#donc nos classes ne sont pas equilibres pour ce probleme car on 5086 cas positif et 558 cas negatifs, donc il faut utiliser une
#metrique comme le score F1, ou la sensibilite ou la precision.
#on n'utilise pas la métrique accuracy car nous donne seulement les TP(true positif) seulement


# In[30]:


#pour afficher ces resultats en % on utilise:
df['SARS-Cov-2 exam result'].value_counts(normalize=True)


# In[21]:


#signification des variables:


# In[31]:


#histogrammes des varibles continues:
#on fait une boucle for en selectionnant toutes les colonnes qui sont des colonnes de types float ds notre dataset
for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])


# In[32]:


#toutes nos courbes sont centrés en 0 et elles ont l'air d'avoir l'ecart type=1 ce qui nous laisse de dire que ces donnees en 
#ete standardises.
#pas mal de variables suivent une distribution normale mais pas toutes.


# In[33]:


#maintenant qu'on a compris le sens des variables de type float.
#on va travailler sur une autre variable qui est l'age:
#on peut commencer par faire un distplot de cette variable:
sns.distplot(df['Patient age quantile'])


# In[34]:


df['Patient age quantile'].value_counts()


# In[35]:


#il nous reste a visualiser les variables categorielles(qualitatives)de type object,on va tout simplement vérifier 
# les différents catégories qui réside dans chaque variable par exemple pour notre variable target on a 2 catégories c'est 
#la catégorie positif et la catégorie négatif.
# pour visualiser on va utiliser la fct de numpy c'est np.unique et cette elle est aussi disponible dans pandas
df['SARS-Cov-2 exam result'].unique()


# In[36]:


for col in df.select_dtypes('object'):
    print(f'{col:-<50} {df[col].unique()}')


# In[37]:


#ensuite on peut compter le nombre de valeur qu'il y a dans chaque catégorie:
for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()


# In[38]:


#clc:variable qualitative : binaire(0,1),viral,Rhinovirus qui semble tres éleveé


# In[39]:


# on passe à l'étape de visualiser la reletion entre variables et target:
#on creér un sous-ensembles positifs et négatifs(individus positifs et individus négatifs)pour faciliter l'analyse:


# In[40]:


positive_df=df[df['SARS-Cov-2 exam result']=='positive']


# In[32]:


negative_df=df[df['SARS-Cov-2 exam result']=='negative']


# In[33]:


#création des sous ensembles blood et viral puisque on avait clairement identifier ces 2 catégorie de variable, en plus on peut 
#faire ca en utilisant tout simplement des petits calcul sur les nombres de valeurs manquates qu'on a dans chaque variable et on 
#identifier que blood 89% de valeur manquates et viral 67%.
#le taux des valeurs manquates est:
df.isna().sum()/df.shape[0]
#on va l'enregistrer dans une variable qui s'appelle missing_rate:
missing_rate=df.isna().sum()/df.shape[0]


# In[34]:


#donc on peut filtrer notre dataset en écrivant par exemple tous les cas missing_rate qui < à 0.9 mais également tous les cas
#de missing_rate qui sont supérieur à 0.88:
blood_columns=df.columns[(missing_rate<0.9) & (missing_rate>0.88)]
blood_columns


# In[35]:


viral_columns=df.columns[(missing_rate<0.88) & (missing_rate>0.75)]


# In[36]:


# visualiser la relation target/blood:
for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()


# In[37]:


#les taux de monocytes,platelets,leukocytes semblent liés au covid-19 car elles ont un taux positif élevé par rapport au négatif
#->hypothese a tester


# In[38]:


#relation target/age:
sns.countplot(x='Patient age quantile',hue='SARS-Cov-2 exam result',data=df)


# In[39]:


#on obtient un graphique dans lequelle on compte le nombre d'apparition de chaque patient age quantile pour les résultats positif
#et negatif de la variable SARS-Cov-2 exam result.


# In[40]:


#relation target/viral(variables qualitatives autrement dit les variables qui sont les test viraux).
#pour comparer deux catégorie ensemble puisque target est une catégorie et test viraux sont des catégories on utilise en statistique
#ce qu'on appelle une crosstab de pandas:
pd.crosstab(df['SARS-Cov-2 exam result'],df['Influenza A'])


# In[41]:


#pour automatiser cela on va faire une boucle for:
for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')


# In[42]:


#conclusion initiales:
#beaucoup de donnéed manquates(au mieux on garde 20% du dataset)
#2 groupes de données intéressantes(viral,sanguin)


# In[43]:


#analyse plus détaillée:
#dans cette analyse détaillés on va s'interesser à la relation variables/variables


# In[44]:


#relations taux sanguin: blood_data/blood_data:
sns.pairplot(df[blood_columns])


# In[45]:


sns.heatmap(df[blood_columns].corr())


# In[46]:


sns.clustermap(df[blood_columns].corr())


# In[47]:


#blood_data/blood_data:certains variables sont tres corrélées:+0.9=le coeff de corrélation(la surveiller plus tard)


# In[48]:


#on visualiser les relations entre le sang et l'age: blood_data/age
#on va utiliser une nouvelle fonction de seaborn qui est lmplot qui nous permet de visualiser les courbes de regression dans 
#le nuage de point.
for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile',y=col,hue='SARS-Cov-2 exam result',data=df)


# In[49]:


df.corr()['Patient age quantile'].sort_values()


# In[50]:


#donc on observe une faible corrélation entre age et taux sanguins.


# In[51]:


#reletion viral/viral:
#reletion entre Influenza et rapid test:
pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])


# In[52]:


pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])


# In[53]:


#influenza rapid test donne de mauvais résultats, il faudra peut etre la laisser tomber.


# In[54]:


#relation viral/blood_data
#relation maladie/blood_data
#on creer une nouvelle variable est malade pour indiquer si un patient est malade avec une maladie a l'exception du Covid-19
#on elemine les 2 dernier colonnes qui sont rapid test A et rapid test B
df['est malade']= np.sum(df[viral_columns[:-2]]=='detected', axis=1) >=1


# In[55]:


df.head()


# In[56]:


malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]


# In[57]:


for col in blood_columns:
    plt.figure()
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()


# In[58]:


#relation hospitalisation / est malade(ou test sanguins) :


# In[59]:


def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'


# In[60]:


df['statut'] = df.apply(hospitalisation, axis=1)


# In[61]:


df.head()


# In[62]:


for col in blood_columns:
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col], label=cat)
    plt.legend()


# In[63]:


df[blood_columns].count()


# In[64]:


df[viral_columns].count()


# In[65]:


df1 = df[viral_columns[:-2]]
df1['covid'] = df['SARS-Cov-2 exam result']
df1.dropna()['covid'].value_counts(normalize=True)


# In[66]:


df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna()['covid'].value_counts(normalize=True)


# In[67]:


#test de student:


# In[68]:


from scipy.stats import ttest_ind


# In[69]:


positive_df


# In[70]:


balanced_neg = negative_df.sample(positive_df.shape[0])


# In[71]:


def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else :
        return 0


# In[72]:


for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')


# In[73]:


#preprocessing du dataset:c'est l'étape de preparer nos donnes avant de les fournir à la machine pour son apprentissage.
#objectifs:
#1-mettre les donnes dans un format propice au ML
#2-améliorer la performance du modele.


# In[74]:


#on re-creer une copie du notre dataframe df:
df=data.copy()
df.head()


# In[75]:


#on va selectioner deux fils des variables qu'on a pu identifier comme réellement utile blood et viral:

missing_rate=df.isna().sum()/df.shape[0]


# In[76]:


#filter notre colonnes de types sang et de type viral:
blood_columns=list(df.columns[(missing_rate<0.9) & (missing_rate>0.88)])
viral_columns=list(df.columns[(missing_rate<0.88) & (missing_rate>0.75)])


# In[77]:


#on créer une liste de colonnes importantes qui sont age et notre target:
key_columns=['Patient age quantile','SARS-Cov-2 exam result']


# In[78]:


#grace a ces trois sous ensembles on va filtrer notre dataframe  df disant que df = a la liste key colonnes+blood_columns+viral_columns
df = df[key_columns + blood_columns + viral_columns]
df.head()
#on se trouve avec 33colonnes qu'on a avait identifier la derniere fois


# In[81]:


#TrainTest-nettoyage-encodage:
from sklearn.model_selection import train_test_split


# In[82]:


trainset, testset = train_test_split(df, test_size=0.2, random_state=0)


# In[83]:


trainset['SARS-Cov-2 exam result'].value_counts()


# In[84]:


testset['SARS-Cov-2 exam result'].value_counts()


# In[85]:


#on a pas le droit de toucher le testset et on a pas le droit de y toucher, ce sont nos donnes future.


# In[86]:


#passant a l'encodage:
#durant notre analyse on pu voir on examinant les variables qualitative c-a-d tous les variables de types object qu'on a avait en
# fait 4 grandes catégories:la catégorie négative,positive,detected,not detected.
#donc pour encodage on va commencer par creer un dictionnaire qui va relier les catégories à des nombres 0,1

code={'positive':1,
       'negative':0,
       'detected':1,
       'not_detected':0}


# In[87]:


#avec ce dictionnaire on va utiliser la fct map de pandas pour appliquer cette fct a toutes les colonnes de type object,donc on 
#écrit une boucle for:
for col in df.select_dtypes('object'):
        df[col] = df[col].map(code)


# In[88]:


df


# In[89]:


#on observe dans notre df que toutes les variables qualitatives ont été remplacé par des 0 et 1.


# In[90]:


df.dtypes.value_counts()


# In[91]:


#pour faire les choses proprement on va copier dans une fct encodage qui va nous permet dans la suite de traiter aussi bien notre
#trainset et notre testset avec cette fct :

def encodage(df):
    code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}
    
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
    return df


# In[92]:


def feature_engineering(df):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1 # suffit que la personne  aurait au moins une maladie
    df = df.drop(viral_columns, axis=1) #on elimine toute les variables du type viral a l'exeption du variable "est malade"
    return df


# In[93]:


### de la meme maniere on va creer une fct imputation pour eliminer notre variable manquantes, dans laquelle on va passer un df 
#que ce soit un trainset ou testset et dans cette fct pour le moment le plus simple de returner dropna(axis=0):

def imputation(df):
   # df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return  df


# In[94]:


#pour finir on va donc creer une fct preprocessing dans on va dire que df=encodage(df), et on va finir cette fct en creant le X 
#et le y dont_on a besoin pour faire le preprocessing on aura a la fois x_train lorsqu'on fera passer le trainset et le x_test
#lorqu'on fera passer le testset

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


# In[97]:


X_train


# In[100]:


X_test, y_test = preprocessing(testset)


# In[101]:


#modélisation:

#notre dataset est pret on peut creer un modele:
#on un modele d'arbre de decision.
#on importe depuis le module tree de sklearn l'estimator DecisionTreeClassifier:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA


# In[102]:


#on creer un model d'arbre de decision dans lequelle je vais fixer le générateur aléatoire sur 0:
#pour eliminer les messages d'avertissement on ajoute un include_bias=False:
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))


# In[103]:


RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))#on ajoute, entre preprocessor et svm, on ajoute une
#opération de normalisation 
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
#on laisse nos modele sur leur hyperparametres de base.


# In[104]:


#on ajoute notre dictionnaire qui contient les differentes modeles:
dict_of_models = {'RandomForest': RandomForest,
                  'AdaBoost' : AdaBoost,
                  'SVM': SVM,
                  'KNN': KNN
                 }


# In[105]:


#métrique d'évaluation:
#puiqu'on parle d'évaluaton il est maintenant temps de creer une procédure d'évaluation qui soit robuste,claire et informative.
#dans cette procedure on va utiliser la métrique F1 parce qu'elle est une bonne metric pour avoir un bonne apercu entre le rapport
#de la precision et de la sensibilité ou la precision et le recall.
#la precision et le recall ce sont des metrics qui permettent de mesurer les proportions d'erreur de type 1 et d'erreur de type 2
#ds notre dataset c-a-d les proportions de faux positive et de faux negative que notre modele effectue.
#on importe aussi confusion_matrix pour voir ses rapport entre faux positive et faux negative.
#et on va importer un rapport de classification est outil qui va nous faire un bilan du recall du précision, score f1 et meme de
#accuracy.
#on importe depuis le module model_selection la fct learning_curve de sklearn par ce que ca va nous etre utile pour comprendre 
#est ce notre model est en overfitting ou en underfitting.

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


# In[108]:


#on va creer un fct evaluation ds laquelle on va passer notre model, ds cette fct on va commencer par entrainer le modele sur les
#donnes X_train,y_train , on suite on calculer les predictions de notre model sur le testse, et on va afficher à l'écran notre 
#matrice de confusion entre y_test et nos prediction ypred, et on va afficher à l'écran notre rapport de classification entre
#y_test et nos prediction ypred.
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


# In[109]:


#on va evaluer tous ces modeles dans une boucle for, avec notre procedure d'evaluation evaluation:
for name, model in dict_of_models.items():
    print(name)
    evaluation(model)
    
#on compare notre modeles, et on va se concentere sur la principale qui est le recall sur notre classe 1(des gens qui tester 
#positif) ou aussi pour etre plus general on va s'interesser au f1-score de la classe 1, et va tenter de prendre le meilleur
#f1-score pour la classe 1 parmi tous nos modeles, et aussi en analysant nos courbes d'apprentissage on constatant que le
#SVM nous indique qu'il n'est plus en overfitting car son score sur le train a chuté mais se rapproche du validation score 
#donc on peut pas parler d'overfitting ds ce cas, donc SVM peut etre tres bon.
#pour le KNN le train score a chuté mais c'est pas grave ce qui est important c'est d'avoir un écart réduit entre le train et
#le validation score par ce que nous montre que le modele a bien appris mais qu'il est capable de généraliser n'a pas 
#overfit son trainset.
#donc on se concentre sur le SVM ou le adaBoost.


# In[110]:


#OPTIMISATION:

#ce qu'on va faire on importe GridSearchCV:

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[111]:


#donc on va commencer à optimiser un modele, donc on va essayer a optimiser le modele SVM, et on peut aussi optimiser le modele
#adaBoost:
SVM


# In[112]:


#lorque on utilise GridSearchCV il faut qu'on defenissent un dictionnaire  d'hyperparametres:
hyper_params = {'svc__gamma':[1e-3, 1e-4],
                'svc__C':[1, 10, 100, 1000], 
                'pipeline__polynomialfeatures__degree':[2, 3],
                'pipeline__selectkbest__k': range(40, 60)}


# In[113]:


#ce dictionnaire d'hyperparams on va passer ds GridSearchCV, ds le scoring on peut prendre recall ou le f1-score, donc nous on va 
#essayer d'optimiser le recall:
#on peut utiliser l'optimisateur RandomizedSearchCV à la place de GridSearchCV , qu'est ce que va faire cette optimisateur c'est
#de chercher de facon aléatoire différentes combinaisant ds tous le domaine des hyperparams ds tous le domaine que nous allons
#définir ds le dictionnaire hyper_params.
#n_iter:c'est a dire combien de fois est ce que cette algorithm RandomizedSearchCV va devoir effectuer une recherche aléatoire 
#avec toutes nos combinaisons, donc combient de combinaisant on y teste aleatoire.

grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4,
                          n_iter=40)
#on entraine notre grid:
grid.fit(X_train, y_train)

#on va afficher les meilleur hyperparams:
print(grid.best_params_) #best_params_ est un attribut de la classe GridSearchCV.

#on peut calculer un vecteur prediction:
y_pred = grid.predict(X_test)

#on compare ce resultat avec les valeurs attendus ds classification_report:
#print(classification_report(y_test, y_pred))


# In[106]:


#donc on reussi a améliorer un petit peu la performance de notre modele en touchant a ses hyperparams, tq le recall et le f1-score
#sont ameliorer.


# In[107]:


#a présent on peurrait passer ce modele ds notre fct evaluation():
evaluation(grid.best_estimator_)


# In[ ]:


#on peut utiliser l'optimisateur RandomizedSearchCV à la place de GridSearchCV , qu'est ce que va faire cette optimisateur c'est
#de chercher de facon aléatoire différentes combinaisant ds tous le domaine des hyperparams ds tous le domaine que nous allons
#définir ds le dictionnaire hyper_params


# In[108]:


#on va finaliser la création de notre modele en observant les courbes precision(recall) et en définissant un seuil de decision
#boundry un seuil de prédiction pour notre modele.
#on va utiliser la fct precision_recall_curve du module metrics qui est une fct qui permet de visualiser la futur precision ou
#bien la futur sensibilité de notre modele en fct d'un seuil de prediction que nous allons définir.

from sklearn.metrics import precision_recall_curve


# In[109]:


#en effet la plupart des modeles qu'on developpe en ML comme le modele SVM ont une _decision_function:

precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))


# In[110]:


plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()


# In[111]:


#on va définir une derniere fct, une fct qui va englober notre modele ca va etre notre veritable fct de prediction,cette fct va 
#retourner le resultat entre model.decision _function(X) sur les donnes X qu'on lui passe,et on va dire que ca nous return true
#si en effet cela est > threshold, sinon false si la fct decision_function(X) est < threshold
def model_final(model, X, threshold=0):
    return model.decision_function(X) > threshold


# In[112]:


y_pred = model_final(grid.best_estimator_, X_test, threshold=-1)


# In[113]:


#pour terminer notre evaluation, on va evaluer y_pred avec score f1:
f1_score(y_test, y_pred)


# In[114]:


#on va ce que va nous donner avec le recall 
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


# In[ ]:


#on a pas d'erreur sur 13 cas negative dans notre testset.
#on pu identifier correctement 11 personnes en disant vous n'etes pas positives vous etes negatives et on a fait 2 erreurs , 2 
#personnes qui ont ete sensé etre negative on l'avait dans l'hopiltal par ce qu'il est problement d'avoir le virus.
#nous une precsion de 92% et un recall 85% pour les gens negatives.
#parmi 3 cas ds notre testset ont ete sensé etre positif , on pu identifier 2, ce qui nous donne un recall de 67% c-a-d on a pu 
#identifier 67% des cas positives avec ce model de ML.
#arreter en 20:00
#on obtient les courbes d'apprentissage pour notre modele et on peut commencer a faire le diagnostic de notre modele.
#ds cas on vois que notre modele est en overfitting.parce que le modele a un score de 100% sur les donnees du trainset et il a 
#parfaitement appris sa lecon, par contre il est incapable de generaliser sur de nouveau cas qu'on lui montre des donnees du 
#validation set(des donnees qu'il n'a pas vu durant son entrainement) on obtient une performance f1 moins bonne.
#a partir de la on peut donc essayer de modifier notre dataset pour lutter contre cette overfitting, par exemple on peut commencer
#par fournir plus de donnees a la machine.
#au lieu d'utiliser un dropna qui nous donne tres peu de donnes, on va modifier le fct imputation avec d'autres choses, et c'est
#la on commence de faire du preprocessing de facon intelligente, on commence par un preprocessing plus basic et on teste des idees
#au fur a mesure pour ameliorer la performance de notre modele et c'est comme ca on fait du preprocessing.
#donc a la place d'utiliser dropna, on utilise a la place un fillna() donc de remplir toutes les valeurs manquantes de notre
#dataset avec une valeur extreme par exemple -999, donc on obtient aucun donnes n'a ete eliminer.
#donc faire fillna(-999) ca marche pas !!!!
#alors a ce stade on pourrait tenter autes chose, on laisse le fillna() et on ajoute une variable qui nous indique la présence 
#de valeur manquantes avec le fct missing_indicator de sklearn.
#on utiliser une autre idee pour lutter contre overfitting c'est selection de variable.


# In[ ]:


#on utilise l'attribut feature_importance qui nous dit quelle sont les variables les plus importante ds la découpe de l'arbre de
#decision:
model.feature_importances_
#27:00


# In[ ]:


#ce tableau on va l'injecter ds un dataframe pandas ds lequlle pour chaque valeur on aura la colonnes associées:
#pour index on va prendre les colonnes de notre trainset
pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12, 8))

#ce graphic nous montre ou sont les variables les plus importantes pour notre arbre de decision.
#pour notre modele les variables de types sang qui sont tres importantes.


# In[ ]:


#a partir de ce graphique on peut faire 2 choses.
# la 1ere est de definir un seuil en dessus du quelle les variables ne sont pas selectionner, par un seuil de 0.01 , les variables
#inferieure a ce seuil ne sont selectionner.
#les variables qui n'ont aucun importances sont les variables de type viral , donc on peut supprimer ces variables.
#qui dit plus de donnees dit une potentiel augmentation du validation score.


# In[ ]:


#on commence par eliminer toutes les variables de types viral.
#on revient au debut a la creation des sous ensembles et elimine viral_columns et on se retrouve avec key_columns et blood_columns.


# In[ ]:


#l'idee d'eliminer les variables n'a pas reussi d'éliminer l'overfitting.
#donc on va essayer d'utiliser un modele régulariser, un modele qui lutte contre overfitting,et un bon exemple c'est le random
#forest, au lieu d'utiliser un arbre de decision on utilise un random forest 
#34:00


# In[ ]:


#ds notre chaine de preprocessing on va ajouter une fct qu'on va appeler feature_engeniering, puisque c'est ca qu'elle s'appelle
#créer des variables a partir des variables a partir des variables deja existante


# In[ ]:




