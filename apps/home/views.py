# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.db.models import Count
from django.http import JsonResponse
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from .models import Banque
import tensorflow as tf
import pandas as pd
import os
import base64
import io
import mysql.connector
path = "apps/home/Ananlyse_exploratoire/fraud_model.h5"
model = tf.keras.models.load_model(path)
from sqlalchemy import create_engine
db_params = {
    "user": "root",
    "password": "",
    "host": "127.0.0.1",
    "database": "dash"
}
engine = create_engine(f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['database']}")
query = "SELECT * FROM home_banque"
new_data = pd.read_sql(query, engine)
print(new_data)

X_new = new_data.drop(["Class","id"], axis=1)  # Assurez-vous d'ajuster le nom de la colonne de l'étiquette de fraude
y_true = new_data["Class"]
print("col", X_new)
print("pred", y_true)
print(X_new.shape)
y_pred_proba = model.predict(X_new)
print(y_pred_proba)
y_pred = (y_pred_proba).astype(int)
print(y_pred)
# Calcul de l'accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Affichage du rapport de classification (précision, rappel, f1-score, etc.)
print("Classification Report:")
print(classification_report(y_true, y_pred))

#########################
accuracy = accuracy_score(y_true, y_pred) * 100
accuracy_str = f"{accuracy:.2f}%"

# Générer le rapport de classification
classification_report_text = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
#######################
#######################
#########################
def generate_confusion_matrix_figure(y_true, y_pred):
    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Créer une représentation de la matrice de confusion avec Seaborn
    plt.figure(figsize=(5.5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non Fraude", "Fraude"], yticklabels=["Non Fraude", "Fraude"])
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies Étiquettes")

    # Convertir la figure en une chaîne base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    #plt.show()
    plt.close()

    data = base64.b64encode(buf.read()).decode('utf-8')

    return data
def generate_true_vs_pred_curve(y_true, y_pred_proba):
    # Créer une représentation de l'évolution de y_true par rapport à y_pred avec Matplotlib
    plt.figure(figsize=(5.5, 3))
    plt.plot(y_true, label='Vraies Étiquettes (y_true)', marker='o')
    plt.plot(y_pred_proba, label='Prédictions (y_pred_proba)', marker='x')
    plt.xlabel('Observations')
    plt.ylabel('Valeurs')
    plt.title('Évolution de y_true par rapport à y_pred')
    plt.legend()

    # Convertir la figure en une chaîne base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    data = base64.b64encode(buf.read()).decode('utf-8')

    return data

# = Banque.objects.values()  # Assurez-vous d'ajuster selon votre modèle Django
# Charger les données à partir de la base de données Django
query = Banque.objects.values()
data_from_db = pd.DataFrame.from_records(query)

# Supprimer les valeurs nulles si nécessaire
data_from_db = data_from_db.dropna()

# Sélectionner les caractéristiques nécessaires
X_new = data_from_db.drop(["Class", "id"], axis=1)  # Ajustez en fonction de votre modèle

# Prédire les probabilités avec votre modèle TensorFlow
y_pred_proba = model.predict(X_new).flatten().tolist()

# Récupérer les vraies valeurs (y_true) depuis la base de données
y_true_from_db = data_from_db["Class"].tolist()

# Maintenant, y_true_from_db contient les vraies valeurs sous forme de liste
print(y_true_from_db, y_pred_proba)

# Générer la courbe d'évolution de y_true par rapport à y_pred
true_vs_pred_curve_figure = generate_true_vs_pred_curve(y_true_from_db, y_pred_proba)
###############################nombre de zero

######################################nombre de zero


#Intervalle de confiance


# Exemple d'utilisation avec des données aléatoires
x_random = np.random.randint(0, 10, 10)
y_random = x_random + np.random.normal(0, 1, 10)

###############################
from django.db.models import Max
montant_max = Banque.objects.aggregate(max_montant=Max('Amount'))['max_montant']

#y_true = [0, 1, 1, 0, 1, 0, 0, 1]
#y_pred_proba = [0.1, 0.8, 0.6, 0.2, 0.7, 0.3, 0.4, 0.9]

#y_true = [0, 1, 1, 0, 1, 0, 0, 1]
# y_pred = [0, 0, 1, 0, 1, 1, 0, 1]
# print(, y_pred)

confusion_matrix_figure = generate_confusion_matrix_figure(y_pred, y_true_from_db)

weights = model.get_weights()
# for i, layer_weights in enumerate(weights):
#     print(f"\nLayer {i + 1} weights:")
#     print(layer_weights)
################################
# Utilisez la fonction aggregate pour compter les occurrences de chaque classe
counts = Banque.objects.values('Class').annotate(count=Count('Class'))

# Créer une réponse JSON pour renvoyer les résultats
response_data = {'fraud_count': 0, 'non_fraud_count': 0}

for entry in counts:
    if entry['Class'] == 1:
        response_data['fraud_count'] = entry['count']
    elif entry['Class'] == 0:
        response_data['non_fraud_count'] = entry['count']

# Générer l'histogramme avec Matplotlib
labels = ['Fraude', 'Non-Fraude']
values = [response_data['fraud_count'], response_data['non_fraud_count']]

plt.bar(labels, values)
plt.xlabel('Classe')
plt.ylabel('Nombre')
plt.title('Nombre de fraudes et de non-fraudes')

# Convertir l'histogramme en image base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
#plt.show()
data1 = base64.b64encode(buffer.read()).decode('utf-8')

############################
#############################confiance

# #############################confiance
#oooooooooooooooooooooooo

data = Banque.objects.all()



# Exemple d'utilisation avec des données de votre modèle
col_to_plot = data.values_list('Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10', 'V11', 'V12', 'V13', 'V14','V15','V16', 'V17', 'V18', 'V19', 'V20','V21','V22','V23','V24','V25','V26','V27','V28', 'Amount')

# Calculer les paramètres de la loi normale
mean = np.mean(col_to_plot)
mean1 = f"{mean:.2f}"
std_dev = np.std(col_to_plot)

# Créer un vecteur x pour représenter les valeurs possibles
x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)

# Calculer les valeurs de la loi normale pour chaque point de x en utilisant les paramètres du modèle
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# Créer un graphique de la loi normale
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Normal Distribution', color='blue')
plt.title('Normal Distribution based on Database Data')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend()

# Convertir le graphique en image base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
data3 = base64.b64encode(buffer.read()).decode('utf-8')
#ooooooooooooooooooooooo



from .models import Client
from django.views import View
from django.shortcuts import get_object_or_404, render
from django.views.generic import TemplateView
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Client

class Client1(TemplateView):
    template_name = "client.html"


class ValidationView(View):
    clients = Client.objects.all()

    def post(self, request):
        accepted_count = int(request.POST.get('accepted_count', 0))
        refused_count = int(request.POST.get('refused_count', 0))

        # Vérifier si le bouton "Valider" a été cliqué
        if 'accept_refuse' in request.POST:
            choice = request.POST.get('accept_refuse')

            if choice == 'accept':
                # Logique pour accepter la transaction
                accepted_count += 1
            elif choice == 'refuse':
                # Logique pour refuser la transaction
                refused_count += 1

        context = {'accepted_count': accepted_count, 'refused_count': refused_count}
        return render(request, 'home/index.html', context)

@login_required(login_url="/login/")
def index(request):
    # Récupérer les clients depuis la base de données
    clients = Client.objects.all()

    # Boucle pour obtenir les valeurs prédites et les imprimer


    # Autres données que vous souhaitez inclure dans le contexte
    nombre_lignes = Banque.objects.count()
    context = {
        'predicted_values': predicted_values,
        'prediction':y_pred_proba,
        'segment': 'index',
        'nombre_lignes': nombre_lignes,
        'dataa': confusion_matrix_figure,
        'data': true_vs_pred_curve_figure,
        'montant': montant_max,
        'performance': accuracy_str,
        'data1': data1,
        'data3': data3,
        'clients': clients,  # Ajoutez les clients au contexte
        'accepted_count': 0,  # Initialisez la variable accepted_count
        'refused_count': 0,  # Initialisez la variable refused_count
        'moyenne':mean1,
        'Ecart-type': std_dev
    }
    return render(request, 'home/index.html', context)


predicted_values = []
@login_required(login_url="/login/")
def pages(request):


    for prediction in y_pred_proba:
        value = int(prediction)  # Convertissez la prédiction en entier si nécessaire
        predicted_values.append(value)
        print(value)

    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template
        form = Banque.objects.all()
        context['f'] = form


        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))

