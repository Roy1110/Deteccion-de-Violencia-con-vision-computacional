import cv2
import numpy as np
import pandas as pd
import joblib
import os
from ultralytics import YOLO
from itertools import combinations
from sklearn.svm import SVC

# --- CONFIGURACIÓN ---
MODEL_PATH = "/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados/modelos_guardados/xgboost.pkl"

# 1. Carga de Modelos
print("Cargando YOLOv8 Nano (CPU)...")
model = YOLO('yolov8n-pose.pt') 

print(f"Cargando clasificador desde: {MODEL_PATH}")
try:
    rf_model, scaler = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"ERROR al cargar modelo: {e}")
    exit()

# 2. Funciones Matemáticas
def angulo_3p(a, b, c):
    ba = a - b
    bc = c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def agregar_estadisticos(df):
    estad = {}
    df = df.fillna(0) 
    for col in df.columns:
        estad[f"{col}_mean"] = df[col].mean()
        estad[f"{col}_std"] = df[col].std()
        estad[f"{col}_max"] = df[col].max()
        estad[f"{col}_min"] = df[col].min()
    return {k: (v if pd.notna(v) else 0) for k, v in estad.items()}

# 3. Inicializar Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30 

# 4. Parámetros de Análisis
segment_duration_seconds = 1
segment_frames = int(fps * segment_duration_seconds)

prev_person_data = {} 
buffer_interacciones_features = []
prediccion_actual = 0  
frame_count = 0

#Variables para Histéresis
consecutive_violence_count = 0
VIOLENCE_THRESHOLD_FRAMES = 3  # Necesita detectar violencia 3 veces seguidas para activar la alerta
ESTADO_FINAL = "Normal" # El estado que se muestra en pantalla

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)

    # Inferencia YOLO
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    frame_con_deteccion = results[0].plot()

    result = results[0]
    keypoints_tensor = result.keypoints.xy if result.keypoints is not None else None
    boxes = result.boxes
    
    if keypoints_tensor is not None and boxes is not None and boxes.id is not None:
        keypoints_np = keypoints_tensor.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()

        current_person_data = {} 
        frame_interaction_results = [] 

        # Extracción de datos individuales
        for i, track_id in enumerate(track_ids):
            puntos_xy = keypoints_np[i]
            if np.all(puntos_xy == 0): continue

            centro = np.mean(puntos_xy, axis=0)

            # Angulos
            try:
                ang_brazo_der = angulo_3p(puntos_xy[6], puntos_xy[8], puntos_xy[10])   
                ang_brazo_izq = angulo_3p(puntos_xy[5], puntos_xy[7], puntos_xy[9])
                ang_pierna_der = angulo_3p(puntos_xy[12], puntos_xy[14], puntos_xy[16])
                ang_pierna_izq = angulo_3p(puntos_xy[11], puntos_xy[13], puntos_xy[15])
            except:
                ang_brazo_der = ang_brazo_izq = ang_pierna_der = ang_pierna_izq = 0

            orientacion = np.degrees(np.arctan2(puntos_xy[5, 1] - puntos_xy[6, 1], puntos_xy[5, 0] - puntos_xy[6, 0] + 1e-6))

            # Velocidad/Aceleración
            vel_media = 0.0
            aceleracion = 0.0
            
            if track_id in prev_person_data:
                prev_pts = prev_person_data[track_id]["puntos_xy"]
                # Calcular desplazamiento promedio de todos los keypoints
                desplazamiento = np.mean(np.linalg.norm(puntos_xy - prev_pts, axis=1))
                vel_media = desplazamiento # Pixeles por frame
                prev_vel = prev_person_data[track_id].get("velocidad_media", 0.0)
                aceleracion = abs(vel_media - prev_vel)

            current_person_data[track_id] = {
                "puntos_xy": puntos_xy, "centro": centro,
                "velocidad_media": vel_media, "aceleracion": aceleracion,
                "ang_brazo_der": ang_brazo_der, "ang_brazo_izq": ang_brazo_izq,
                "ang_pierna_der": ang_pierna_der, "ang_pierna_izq": ang_pierna_izq,
                "orientacion": orientacion
            }

        # Interacciones
        ids = list(current_person_data.keys())
        if len(ids) >= 2:
            for id_a, id_b in combinations(ids, 2):
                d1, d2 = current_person_data[id_a], current_person_data[id_b]
                
                distancia = np.linalg.norm(d1["centro"] - d2["centro"])
                velocidad_rel = abs(d1["velocidad_media"] - d2["velocidad_media"])
                acel_prom = np.mean([d1["aceleracion"], d2["aceleracion"]])

                
                if distancia > 150: 
                    continue 

                # 2. Filtro de Movimiento
                if velocidad_rel < 3.5 and acel_prom < 1.0:
                    continue

                features = {
                    "distancia_centros": distancia,
                    "velocidad_relativa": velocidad_rel,
                    "aceleracion_prom": acel_prom,
                    "diferencia_orientacion": abs(d1["orientacion"] - d2["orientacion"]) % 360,
                    "ang_brazo_der": d1["ang_brazo_der"], "ang_brazo_izq": d1["ang_brazo_izq"],
                    "ang_pierna_der": d1["ang_pierna_der"], "ang_pierna_izq": d1["ang_pierna_izq"],
                    "movimiento_relativo": np.dot((d2["centro"] - d1["centro"]) / (np.linalg.norm(d2["centro"] - d1["centro"]) + 1e-6), d2["centro"] - d1["centro"])
                }
                frame_interaction_results.append(features)

        prev_person_data = current_person_data
        buffer_interacciones_features.extend(frame_interaction_results)

    # Prediccion cada segundo
    if frame_count % segment_frames == 0:
        prediccion_raw = 0
        
        if len(buffer_interacciones_features) > 0:
            try:
                df = pd.DataFrame(buffer_interacciones_features)
                stats = agregar_estadisticos(df)
                
                # Asegurar orden de columnas
                feat_names = scaler.get_feature_names_out() if hasattr(scaler, "get_feature_names_out") else scaler.feature_names_in_
                X_df = pd.DataFrame([stats]).reindex(columns=feat_names).fillna(0)
                
                # Transformar
                X_input = scaler.transform(X_df)
                
                # Predecir
                prediccion_raw = rf_model.predict(X_input)[0]
                

            except Exception as e:
                print(f"Error predicción: {e}")
        
        # Lógica de Estado
        if prediccion_raw == 1:
            consecutive_violence_count += 1
        else:
            consecutive_violence_count = 0 # Reinicia si hay un momento de calma

        # Solo activamos la alarma si hay violencia sostenida
        if consecutive_violence_count >= VIOLENCE_THRESHOLD_FRAMES:
            ESTADO_FINAL = "VIOLENCIA DETECTADA"
            color_estado = (0, 0, 255)
        else:
            ESTADO_FINAL = "Normal"
            color_estado = (0, 255, 0)

        print(f"Predicción Raw: {prediccion_raw} | Contador: {consecutive_violence_count} | Estado: {ESTADO_FINAL}")
            
        buffer_interacciones_features = [] # Reset buffer

    # MOSTRAR EN PANTALLA
    # Usamos ESTADO_FINAL en lugar de prediccion_actual directa
    height, width = frame_con_deteccion.shape[:2]
    cv2.rectangle(frame_con_deteccion, (0, 0), (width, 60), (0, 0, 0), -1)
    
    color_texto = (0, 0, 255) if ESTADO_FINAL == "VIOLENCIA DETECTADA" else (0, 255, 0)
    
    cv2.putText(frame_con_deteccion, f"Estado: {ESTADO_FINAL}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)

    cv2.imshow('Sistema Deteccion Violencia', frame_con_deteccion)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()