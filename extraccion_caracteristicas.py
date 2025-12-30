import os
import cv2
import numpy as np
import pandas as pd
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import time


#   FUNCIÓN: cargar modelo dentro de cada proceso
def cargar_modelo_keypoints_cpu():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


#   PROCESAMIENTO INDIVIDUAL DE UN VIDEO
def procesar_video(args):
    categoria, ruta_video, salida_temp, salto_frames = args
    nombre_video = os.path.basename(ruta_video)
    temp_csv = os.path.join(salida_temp, f"{nombre_video}.csv")

    try:
        predictor = cargar_modelo_keypoints_cpu()  # modelo dentro del proceso
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resultados = []
        frame_anterior = {}

        for frame_idx in range(0, total_frames, salto_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            outputs = predictor(frame)
            instancias = outputs["instances"].to("cpu")

            if not instancias.has("pred_keypoints"):
                frame_anterior = {}
                continue

            keypoints = instancias.pred_keypoints.numpy()
            for pid, puntos in enumerate(keypoints):
                puntos_xy = puntos[:, :2]
                if pid in frame_anterior and frame_anterior[pid].size != 0:
                    prev_pts = frame_anterior[pid]
                    desplazamiento = np.linalg.norm(puntos_xy - prev_pts[:, :2], axis=1)
                    velocidad = np.mean(desplazamiento)
                else:
                    velocidad = 0.0

                fila = [categoria, nombre_video, frame_idx, pid, round(velocidad, 4)]
                for i in range(17):
                    fila += [round(puntos_xy[i][0], 2), round(puntos_xy[i][1], 2)]
                resultados.append(fila)

            frame_anterior = {pid: puntos for pid, puntos in enumerate(keypoints)}

        cap.release()

        if resultados:
            columnas = ["categoria", "video", "frame", "persona_id", "velocidad_media"] + \
                       [f"x_{i}" for i in range(17)] + [f"y_{i}" for i in range(17)]
            df = pd.DataFrame(resultados, columns=columnas)
            df.to_csv(temp_csv, index=False)

        return nombre_video

    except Exception as e:
        print(f"Error procesando {nombre_video}: {e}")
        return None


#   PROCESAMIENTO GLOBAL EN PARALELO
def procesar_en_paralelo(ruta_base, salida_csv, salto_frames=10):
    categorias = ["violencia_limpios", "noViolencia"]
    salida_temp = os.path.join(ruta_base, "salida_deteccion/temp_videos")
    os.makedirs(salida_temp, exist_ok=True)

    # procesos a usar
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    max_proc = min(cpu_count(), max(1, int(total_ram // 3)))
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Núcleos totales: {cpu_count()}")
    print(f"Memoria total: {total_ram:.1f} GB")
    print(f"Usando hasta {max_proc} procesos.\n")

    # detectar videos ya procesados
    procesados = {os.path.splitext(f)[0] for f in os.listdir(salida_temp) if f.endswith(".csv")}
    print(f"{len(procesados)} videos ya procesados (reanudar desde ahí).\n")

    # generar lista de tareas
    tareas = []
    for categoria in categorias:
        dir_cat = os.path.join(ruta_base, categoria)
        if not os.path.exists(dir_cat):
            continue
        for video in sorted(os.listdir(dir_cat)):
            if video.lower().endswith(".mp4"):
                if os.path.splitext(video)[0] not in procesados:
                    tareas.append((categoria, os.path.join(dir_cat, video), salida_temp, salto_frames))

    print(f"{len(tareas)} videos nuevos por procesar.\n")

    if not tareas:
        print("No hay videos pendientes. Solo combinando CSVs...")
        combinar_csvs(salida_temp, salida_csv)
        return

    inicio = time.time()
    with Pool(processes=max_proc) as pool:
        for _ in tqdm(pool.imap_unordered(procesar_video, tareas),
                      total=len(tareas), desc="Procesando videos", unit="video"):
            pass

    combinar_csvs(salida_temp, salida_csv)
    print(f"\nExtracción completada en {time.time()-inicio:.1f} s")
    print(f"Resultados combinados en: {salida_csv}")


#   COMBINACIÓN FINAL DE CSVs
def combinar_csvs(carpeta_temp, salida_csv):
    archivos = [os.path.join(carpeta_temp, f) for f in os.listdir(carpeta_temp) if f.endswith(".csv")]
    if not archivos:
        print("No se encontraron archivos parciales.")
        return

    print(f"\nCombinando {len(archivos)} archivos parciales...")
    dfs = []
    for f in archivos:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error leyendo {f}: {e}")

    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv(salida_csv, index=False)
    print("CSV combinado guardado.")

if __name__ == "__main__":
    ruta_base = "/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados"
    salida_csv = os.path.join(ruta_base, "vectores_keypoints_full.csv")

    procesar_en_paralelo(ruta_base, salida_csv, salto_frames=10)
