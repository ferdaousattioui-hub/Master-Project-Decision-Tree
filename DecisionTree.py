import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Projet ML - Arbre de Décision Iris", layout="wide")

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez la section :", ["1. Présentation (PDF)", "2. Dashboard Interactive Arbre de Décision"])

# --- CONFIGURATION ESTHÉTIQUE ---
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# --- CHARGEMENT DU DATASET IRIS ---
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris

# --- SECTION 1 : PRÉSENTATION PPT ---
if page == "1. Présentation (PDF)":
    st.title("📂 Présentation du Projet - Arbre de Décision")
    st.write("Voici La Présentation Théorique de l'algorithme des Arbres de Décision.")
    
    # Lien exact 
    file_id = "146j4QhO8Xnmnd6Ky1tHmzClBIbRxEHRW"  
    lien_drive_embed = f"https://drive.google.com/file/d/{file_id}/preview"
    
    st.components.v1.html(
        f'<iframe src="{lien_drive_embed}" style="width:100%; height:750px;" frameborder="0" allowfullscreen></iframe>',
        height=750
    )

# --- SECTION 2 : DASHBOARD INTERACTIVE ARBRE DE DÉCISION ---
elif page == "2. Dashboard Interactive Arbre de Décision":
    st.title("💻 Structure Interactive d'un Arbre de Décision (Iris)")
    st.write("Modifiez la profondeur maximale et les critères mathématiques pour analyser l'élagage et la pureté des nœuds.")

    # --- BARRE LATÉRALE DES PARAMÈTRES (INTERACTIF) ---
    st.sidebar.subheader("🎛️ Hyperparamètres de l'Arbre")
    criterion_choice = st.sidebar.selectbox("Critère de Scission (Pureté)", ["gini", "entropy", "log_loss"])
    max_depth_value = st.sidebar.slider("Profondeur Maximale (max_depth)", min_value=1, max_value=6, value=3, step=1)
    test_size_ratio = st.sidebar.slider("Proportion du jeu de test (%)", min_value=10, max_value=50, value=30, step=5) / 100.0

    # Chargement et préparation (Exactement kima f script dyalk)
    iris = load_iris_data()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

    # Entraînement en direct avec les curseurs
    clf = DecisionTreeClassifier(criterion=criterion_choice, max_depth=max_depth_value, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred) * 100

    # --- PANNEAU DE STATISTIQUES ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="📊 Critère Sélectionné", value=criterion_choice.upper())
    col2.metric(label="🌳 Profondeur Active", value=f"{max_depth_value} Niveaux")
    col3.metric(label="🎯 Précision sur Test (Accuracy)", value=f"{acc:.2f} %")

    st.markdown("---")

    # --- VISUALISATION : L'ARBRE GÉANT ---
    st.subheader("🌲 Structure Vectorielle Générée de l'Arbre")
    st.write("Ce graphe montre comment les règles logiques IF-THEN découpent récursivement le dataset.")
    
    with plt.style.context('default'):
        fig_tree, ax_tree = plt.subplots(figsize=(14, 8), facecolor='white')
        plot_tree(clf, 
                  feature_names=iris.feature_names, 
                  class_names=iris.target_names, 
                  filled=True, 
                  rounded=True,
                  fontsize=10,
                  ax=ax_tree)
        plt.tight_layout()
        st.pyplot(fig_tree)

    st.markdown("---")

    # --- ANALYSE DES ERREURS ---
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("📊 Matrice de Confusion")
        with plt.style.context('default'):
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5.5), facecolor='white')
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=iris.target_names, yticklabels=iris.target_names, 
                        ax=ax_cm, cbar=False, square=True, annot_kws={"size": 14})
            
            ax_cm.set_title("Matrice de Confusion de l'Arbre", fontsize=12, fontweight='bold', pad=10)
            ax_cm.set_ylabel('Vraie Espèce (Réelle)', fontsize=10)
            ax_cm.set_xlabel('Espèce Prédite', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_cm)

    with col_right:
        st.subheader("📋 Rapport de Performance par Classe")
        report_dict = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        st.dataframe(df_report.style.format(precision=3), use_container_width=True)
