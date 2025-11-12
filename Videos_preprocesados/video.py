import cv2
import numpy as np
import pandas as pd
import torch
import joblib
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from itertools import combinations

# --- Imports adicionales del Script 2 ---
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ... otros imports ...
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# ----------------------------------------

# ================================
# 1️⃣ CONFIGURAR DETECTRON2
# ================================
print("Cargando modelo Detectron2...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Umbral de confianza
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
coco_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# ================================
# 2️⃣ FUNCIONES AUXILIARES (DE TUS SCRIPTS DE ENTRENAMIENTO)
# ================================

def angulo_3p(a, b, c):
    """Calcula el ángulo (en grados) entre tres puntos 2D: a–b–c."""
    ba = a - b
    bc = c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def agregar_estadisticos(df):
    """Calcula agregados (mean, std, max, min) para un dataframe de features."""
    estad = {}
    # Rellenar NaNs antes de calcular estadísticas
    df = df.fillna(0) 
    
    for col in df.columns:
        estad[f"{col}_mean"] = df[col].mean()
        estad[f"{col}_std"] = df[col].std()
        estad[f"{col}_max"] = df[col].max()
        estad[f"{col}_min"] = df[col].min()
    
    # Asegurarse de que no haya NaNs en las estadísticas finales (p.ej. si std es 0)
    return {k: (v if pd.notna(v) else 0) for k, v in estad.items()}


# ================================
# 3️⃣ CARGAR MODELO Y SCALER
# ================================
# ================================
# 3️⃣ CARGAR MODELO Y SCALER
# ================================
print("Cargando modelo de clasificación (RF) y scaler...")
try:
    # --- CAMBIO APLICADO ---
    # Desempaquetar la tupla guardada en el archivo .pkl
    # Asumimos que "modelo_rf.pkl" contiene (modelo, scaler)
    rf_model, scaler = joblib.load("/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados/modelos_guardados/random_forest.pkl")
    
    # Ya no necesitas cargar "scaler.pkl" por separado
    # scaler = joblib.load("scaler.pkl") 
    
except FileNotFoundError:
    print("Error: No se encontró 'modelo_rf.pkl'.")
    exit()
except ValueError:
    print("Error al cargar 'modelo_rf.pkl'.")
    print("Asegúrate de que SÍ contenga una tupla (modelo, scaler).")
    exit()

# ================================
# 4️⃣ CONFIGURAR VIDEO
# ================================
video_path = "//home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados/noViolencia/NV_2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(os.getcwd(), "salida_detectada_full_features6_svm.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"Video de entrada: {video_path} ({width}x{height} @ {fps} FPS)")

# ================================
# 5️⃣ PARÁMETROS DE SEGMENTACIÓN Y ESTADO
# ================================
segment_duration_seconds = 1
segment_frames = int(fps * segment_duration_seconds)
print(f"Procesando en segmentos de {segment_frames} frames ({segment_duration_seconds} seg)...")

# --- Variables de estado para calcular velocidad/aceleración ---
# Guardará los datos de la persona (centro, velocidad) del frame anterior
prev_person_data = {} 
# -----------------------------------------------------------

# Buffer para guardar todas las filas de interacciones del segmento
buffer_interacciones_features = []
prediccion_actual = 0  # 0 = No Violencia, 1 = Violencia

# ================================
# 6️⃣ PROCESAR FRAME A FRAME
# ================================
print("Procesando video (esto puede tardar)...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % fps == 0:
        print(f"  ... procesando segundo {frame_count // fps}")

    outputs = predictor(frame)
    instancias = outputs["instances"].to("cpu")

    # --- Visualización (como antes) ---
    v = Visualizer(frame[:, :, ::-1], metadata=coco_metadata, scale=1.0)
    out_vis = v.draw_instance_predictions(instancias)
    frame_con_deteccion = out_vis.get_image()[:, :, ::-1]
    frame_con_deteccion = np.ascontiguousarray(frame_con_deteccion) # FIX para cv2.putText

    # --- INICIO DE LA LÓGICA DE FEATURES INTEGRADAS ---
    keypoints = instancias.pred_keypoints.numpy() if instancias.has("pred_keypoints") else []
    
    current_person_data = {} # Datos de personas en ESTE frame
    frame_interaction_results = [] # Interacciones de ESTE frame

    # --- Bucle 1: Calcular features individuales (Script 1 y 2) ---
    for pid, puntos in enumerate(keypoints):
        puntos_xy = puntos[:, :2]
        centro = np.mean(puntos_xy, axis=0)

        # Calcular ángulos
        try:
            ang_brazo_der = angulo_3p(puntos_xy[5], puntos_xy[7], puntos_xy[9])   # hombro, codo, muñeca
            ang_brazo_izq = angulo_3p(puntos_xy[6], puntos_xy[8], puntos_xy[10])
            ang_pierna_der = angulo_3p(puntos_xy[11], puntos_xy[13], puntos_xy[15])
            ang_pierna_izq = angulo_3p(puntos_xy[12], puntos_xy[14], puntos_xy[16])
        except Exception:
            ang_brazo_der = ang_brazo_izq = ang_pierna_der = ang_pierna_izq = np.nan

        # Orientación corporal
        orientacion = np.degrees(np.arctan2(puntos_xy[5, 1] - puntos_xy[6, 1], puntos_xy[5, 0] - puntos_xy[6, 0] + 1e-6))

        # Calcular Velocidad y Aceleración (comparando con frame anterior)
        velocidad_media = 0.0
        aceleracion = 0.0
        if pid in prev_person_data:
            prev_data = prev_person_data[pid]
            prev_pts = prev_data["puntos_xy"]
            
            # Calcular velocidad (como en tu script 1)
            desplazamiento = np.linalg.norm(puntos_xy - prev_pts, axis=1)
            velocidad_media = np.mean(desplazamiento)

            # Calcular aceleración (como en tu script 2)
            prev_velocidad = prev_data.get("velocidad_media", 0.0)
            aceleracion = abs(velocidad_media - prev_velocidad)

        current_person_data[pid] = {
            "puntos_xy": puntos_xy,
            "centro": centro,
            "velocidad_media": velocidad_media,
            "aceleracion": aceleracion,
            "ang_brazo_der": ang_brazo_der,
            "ang_brazo_izq": ang_brazo_izq,
            "ang_pierna_der": ang_pierna_der,
            "ang_pierna_izq": ang_pierna_izq,
            "orientacion": orientacion
        }

    # --- Bucle 2: Calcular features de interacción (Script 2) ---
    pids = list(current_person_data.keys())
    if len(pids) >= 2:
        for i, j in combinations(pids, 2):
            d1 = current_person_data[i]
            d2 = current_person_data[j]

            dist_centro = np.linalg.norm(d1["centro"] - d2["centro"])
            vel_rel = abs(d1["velocidad_media"] - d2["velocidad_media"])
            acel_prom = np.mean([d1["aceleracion"], d2["aceleracion"]])
            orient_diff = abs(d1["orientacion"] - d2["orientacion"]) % 360
            
            dir_vec = d2["centro"] - d1["centro"]
            dir_norm = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
            mov_rel = np.dot(dir_norm, d2["centro"] - d1["centro"]) # Revisar esta lógica si da problemas

            # Crear un diccionario de features (fila)
            interaction_features = {
                "distancia_centros": dist_centro,
                "velocidad_relativa": vel_rel,
                "aceleracion_prom": acel_prom,
                "diferencia_orientacion": orient_diff,
                # Usamos los ángulos de la persona 1 (puedes cambiarlos a p2 o promediar si prefieres)
                "ang_brazo_der": d1["ang_brazo_der"],
                "ang_brazo_izq": d1["ang_brazo_izq"],
                "ang_pierna_der": d1["ang_pierna_der"],
                "ang_pierna_izq": d1["ang_pierna_izq"],
                "movimiento_relativo": mov_rel
            }
            frame_interaction_results.append(interaction_features)

    # Actualizar el estado para el próximo frame
    prev_person_data = current_person_data
    
    # Agregar las interacciones de este frame al buffer del segmento
    buffer_interacciones_features.extend(frame_interaction_results)
    # --- FIN DE LA LÓGICA DE FEATURES ---


    # Cada segmento → generar predicción
    if (frame_count % segment_frames == 0) and (len(buffer_interacciones_features) > 0):
        
        # 1. Convertir el buffer de dicts a un DataFrame
        df_segmento = pd.DataFrame(buffer_interacciones_features)
        
        # 2. Aplicar agregación estadística (Script 3)
        features_dict = agregar_estadisticos(df_segmento)
        
        # 3. Convertir a DataFrame de 1 fila para el scaler
        X_df = pd.DataFrame([features_dict])

        try:
            # 4. REVISADO: Preparar datos y predecir según el tipo de modelo
            
            # Asegurarse de que el DataFrame tenga las columnas correctas
            X_df_filled = X_df.reindex(columns=scaler.get_feature_names_out()).fillna(0)

            X_para_predecir = None

            # Comprobar el tipo de modelo cargado en 'rf_model'
            if isinstance(rf_model, (RandomForestClassifier, XGBClassifier)):
                # RF y XGBoost se entrenaron con el DataFrame SIN escalar (X_train)
                # Usamos X_df_filled (DataFrame de Pandas)
                X_para_predecir = X_df_filled
            
            elif isinstance(rf_model, SVC):
                # SVM se entrenó con datos ESCALADOS (X_train_scaled)
                # Escalamos los datos y usamos el array de NumPy
                X_para_predecir = scaler.transform(X_df_filled)
            
            else:
                # Si es un modelo desconocido, intentar como SVM por si acaso
                print("ADVERTENCIA: Tipo de modelo no reconocido, intentando con datos escalados.")
                X_para_predecir = scaler.transform(X_df_filled)

            # 5. Predecir
            if X_para_predecir is not None:
                pred = rf_model.predict(X_para_predecir)[0]
                prediccion_actual = pred
        
        except ValueError as e:
            print(f"Error al predecir: {e}")
            pass # Mantener la predicción anterior
        
        buffer_interacciones_features = []  # limpiar buffer


    # Mostrar etiqueta
    label = "Violencia" if prediccion_actual == 1 else "No violencia"
    color = (0, 0, 255) if label == "Violencia" else (0, 255, 0)
    cv2.putText(frame_con_deteccion, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    out.write(frame_con_deteccion)

cap.release()
out.release()
print(f"\n✅ Video procesado y guardado en: {output_path}")