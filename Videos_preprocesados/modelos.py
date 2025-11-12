import os
import time
import numpy as np
import pandas as pd
import psutil
import multiprocessing as mp
import pickle  # <--- 1. CAMBIO: Usar pickle en lugar de joblib
import json    # <--- 2. AÑADIDO: Para guardar los hiperparámetros
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ==========================================================
# 1. CARGA DE DATOS
# ==========================================================
ruta_csv = "/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados/salida_deteccion/dataset_agregado_local.csv"
print(f"Cargando dataset desde: {ruta_csv}")

df = pd.read_csv(ruta_csv)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

def mapear_categoria(cat):
    cat_lower = str(cat).lower()
    return 0 if "noviolencia" in cat_lower else 1

X = df.drop(columns=["categoria", "video"], errors="ignore")
y = df["categoria"].apply(mapear_categoria)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Datos listos: {len(X_train)} entrenamiento, {len(X_test)} prueba.")

# ==========================================================
# 2. CONFIGURACIÓN ADAPTATIVA
# ==========================================================
cpu_total = mp.cpu_count()
ram_gb = psutil.virtual_memory().total / (1024**3)
procesos = max(2, min(cpu_total // 4, 4))
print(f"CPU totales: {cpu_total} | RAM: {ram_gb:.1f} GB | Usando {procesos} procesos")

# ==========================================================
# 3. FUNCIÓN DE ENTRENAMIENTO
# ==========================================================
def entrenar_modelo(nombre):
    inicio = time.time()
    if nombre == "Random Forest":
        print("\nEntrenando Random Forest...")
        param_grid = {
            'n_estimators': [100, 200,250, 300],
            'max_depth': [10, 15,18, 20, None],
            'min_samples_leaf': [1, 2, 3],
            'min_samples_split': [2],
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42),
                            param_grid, cv=3, scoring='f1', n_jobs=1)
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_

    elif nombre == "XGBoost":
        print("\nEntrenando XGBoost...")
        param_grid =  {
        'n_estimators': [450, 500, 550, 600],
        'max_depth': [7, 8, 9],
        'learning_rate': np.linspace(0.003, 0.01, 5),
        'subsample': np.linspace(0.6, 0.7, 5),
        'colsample_bytree': np.linspace(0.9, 1.0, 5),
        'gamma': [0, 0.05, 0.1, 0.2],
        'reg_alpha': [0, 0.05, 0.1],
        'reg_lambda': [0, 0.5, 1.0]
        }

        grid = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'),
                                  param_grid, cv=3, scoring='f1',
                                  n_iter=20, random_state=42, n_jobs=1)
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_

    elif nombre == "SVM":
        print("\nEntrenando SVM...")
        param_grid = {
            'C': [3, 5, 7, 10], # Centrado en 5
            'gamma': [0.001, 0.003, 0.005, 0.007, 0.01], # Rango muy fino alrededor de 0.005
            'kernel': ['rbf']
        }
        grid = GridSearchCV(SVC(probability=True, random_state=42),
                            param_grid, cv=3, scoring='f1', n_jobs=1)
        grid.fit(X_train_scaled, y_train)
        modelo = grid.best_estimator_

    duracion = (time.time() - inicio) / 60
    best_params = grid.best_params_ # <--- 3. AÑADIDO: Obtener los mejores hiperparámetros
    print(f"{nombre} finalizado en {duracion:.2f} min.")
    
    # <--- 4. CAMBIO: Devolver también los hiperparámetros
    return nombre, modelo, duracion, best_params

# ==========================================================
# 4. ENTRENAMIENTO EN PARALELO
# ==========================================================
modelos = ["Random Forest", "XGBoost", "SVM"]

with mp.Pool(processes=procesos) as pool:
    # 'resultados' ahora tendrá (nombre, modelo, duracion, best_params)
    resultados = pool.map(entrenar_modelo, modelos)

# ==========================================================
# 5. EVALUACIÓN FINAL
# ==========================================================
evaluaciones = []
# <--- 5. CAMBIO: Desempaquetar los best_params
for nombre, modelo, duracion, best_params in resultados:
    print(f"\nEvaluando {nombre}...")
    X_eval = X_test_scaled if nombre == "SVM" else X_test

    y_pred = modelo.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["No violencia", "Violencia"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.show()

    evaluaciones.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
        "Tiempo_min": duracion,
        "Hiperparametros": best_params # <--- 6. AÑADIDO: Guardar hiperparámetros en el dict
    })

# ==========================================================
# 6. GUARDADO DE RESULTADOS (CSV)
# ==========================================================
# El CSV ahora contendrá la columna 'Hiperparametros'
pd.DataFrame(evaluaciones).to_csv("evaluacion_modelos_supervisados.csv", index=False)
print("\nResultados guardados en evaluacion_modelos_supervisados.csv")

# ==========================================================
# 7. GUARDADO DE MODELOS (NUEVA SECCIÓN CON PICKLE)
# ==========================================================
model_dir = "modelos_guardados"
os.makedirs(model_dir, exist_ok=True)
print(f"\nGuardando modelos y scaler en: {model_dir}")

# <--- 7. CAMBIO: Desempaquetar best_params aquí también
for nombre, modelo, duracion, best_params in resultados:
    # Crear un nombre de archivo limpio, ej: "random_forest"
    file_name = nombre.lower().replace(' ', '_')
    
    # --- Guardar el modelo y scaler como tupla usando PICKLE ---
    pkl_path = os.path.join(model_dir, file_name + '.pkl')
    try:
        with open(pkl_path, 'wb') as f: # 'wb' es "write binary"
            pickle.dump((modelo, scaler), f)
        print(f"  -> Modelo guardado: {pkl_path}")
    except Exception as e:
        print(f"  -> ERROR al guardar {pkl_path}: {e}")

    # --- Guardar los hiperparámetros en un JSON separado ---
    param_path = os.path.join(model_dir, file_name + '_params.json')
    try:
        # Convertir tipos numpy a nativos de python si es necesario
        params_serializables = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best_params.items()}
        with open(param_path, 'w') as f: # 'w' es "write text"
            json.dump(params_serializables, f, indent=4)
        print(f"  -> Hiperparámetros guardados: {param_path}")
    except Exception as e:
        print(f"  -> ERROR al guardar {param_path}: {e}")


print("¡Guardado de modelos completo!")