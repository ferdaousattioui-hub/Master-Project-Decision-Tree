#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. Chargement du dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# 3. Séparation en ensemble d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entraînement de l'arbre de décision
# On limite la profondeur à 3 pour que l'arbre reste lisible et éviter l'overfitting
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 5. Prédiction et évaluation du modèle
y_pred = clf.predict(X_test)
print(f"Précision du modèle sur le jeu de test : {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Visualisation de l'arbre
plt.figure(figsize=(15, 10))
plot_tree(clf, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, 
          rounded=True)
plt.title("Visualisation de l'arbre de décision sur le dataset Iris")
plt.show()


# In[ ]:




