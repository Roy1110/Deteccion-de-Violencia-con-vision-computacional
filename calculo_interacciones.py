#!/usr/bin/env python3
# calculo_interacciones.py
# Calcula características extendidas de movimiento e interacción entre personas
# a partir del CSV de keypoints generado por la extracción paralela.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------
# Función auxiliar: ángulo entre tres puntos
# ---------------------------
def angulo_3p(a, b, c):
    """Calcula el ángulo (en grados) entre tres puntos 2D: a–b–c."""
    ba = a - b
    bc = c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


# ---------------------------
# Cálculo de interacciones extendidas
# ---------------------------
def calcular_interacciones_extendido(csv_entrada, csv_salida):
    """
    Calcula características avanzadas de movimiento e interacción entre personas.
    Incluye:
      - Distancias y contacto físico.
      - Velocidad relativa y aceleración.
      - Ángulos de brazos y piernas.
      - Orientación corporal.
      - Dirección del movimiento.
      - Frecuencia de contacto.
    """

    print("Cargando CSV base...")
    df = pd.read_csv(csv_entrada)

    if not {"video", "frame", "persona_id"}.issubset(df.columns):
        raise ValueError("El CSV no contiene las columnas esperadas.")

    os.makedirs(os.path.dirname(csv_salida), exist_ok=True)
    resultados = []

    key_cols_x = [f"x_{i}" for i in range(17)]
    key_cols_y = [f"y_{i}" for i in range(17)]

    # Para calcular aceleraciones, guardamos el frame previo por video y persona
    prev_coords = {}
    prev_vel = {}

    # Recorremos por video y frame
    for (video, frame), grupo in tqdm(df.groupby(["video", "frame"]), desc="Procesando interacciones"):

        categoria = grupo["categoria"].values[0]

        # Saltar si hay menos de 1 persona
        if len(grupo) < 1:
            continue

        # Guardar keypoints y características individuales
        persona_data = {}

        for _, row in grupo.iterrows():
            pid = int(row["persona_id"])
            xs = np.array([row[c] for c in key_cols_x])
            ys = np.array([row[c] for c in key_cols_y])
            puntos = np.stack([xs, ys], axis=1)

            # Centro de masa (promedio)
            centro = np.mean(puntos, axis=0)

            # Ángulos de brazos y piernas (COCO keypoints)
            try:
                ang_brazo_der = angulo_3p(puntos[5], puntos[7], puntos[9])   # hombro, codo, muñeca
                ang_brazo_izq = angulo_3p(puntos[6], puntos[8], puntos[10])
                ang_pierna_der = angulo_3p(puntos[11], puntos[13], puntos[15])
                ang_pierna_izq = angulo_3p(puntos[12], puntos[14], puntos[16])
            except:
                ang_brazo_der = ang_brazo_izq = ang_pierna_der = ang_pierna_izq = np.nan

            # Orientación corporal: vector hombro_izq → hombro_der
            orientacion = np.degrees(np.arctan2(puntos[5, 1] - puntos[6, 1], puntos[5, 0] - puntos[6, 0]))

            # Aceleración: requiere frame anterior
            aceleracion = 0.0
            if (video, pid) in prev_coords:
                prev_centro = prev_coords[(video, pid)]
                desplazamiento = np.linalg.norm(centro - prev_centro)
                prev_v = prev_vel.get((video, pid), 0.0)
                vel_actual = row["velocidad_media"]
                aceleracion = abs(vel_actual - prev_v)
                prev_vel[(video, pid)] = vel_actual
                prev_coords[(video, pid)] = centro
            else:
                prev_coords[(video, pid)] = centro
                prev_vel[(video, pid)] = row["velocidad_media"]

            persona_data[pid] = {
                "centro": centro,
                "velocidad": row["velocidad_media"],
                "aceleracion": aceleracion,
                "ang_brazo_der": ang_brazo_der,
                "ang_brazo_izq": ang_brazo_izq,
                "ang_pierna_der": ang_pierna_der,
                "ang_pierna_izq": ang_pierna_izq,
                "orientacion": orientacion
            }

        # Calcular interacciones entre personas
        pids = list(persona_data.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p1, p2 = pids[i], pids[j]
                d1, d2 = persona_data[p1], persona_data[p2]

                dist_centro = np.linalg.norm(d1["centro"] - d2["centro"])
                vel_rel = abs(d1["velocidad"] - d2["velocidad"])
                acel_prom = np.mean([d1["aceleracion"], d2["aceleracion"]])
                orient_diff = abs(d1["orientacion"] - d2["orientacion"]) % 360

                # Movimiento relativo (si se acercan o alejan)
                dir_vec = d2["centro"] - d1["centro"]
                dir_norm = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
                mov_rel = np.dot(dir_norm, d2["centro"] - d1["centro"])

                resultados.append([
                    categoria, video, frame,
                    p1, p2,
                    round(dist_centro, 2),
                    round(vel_rel, 3),
                    round(acel_prom, 3),
                    round(orient_diff, 2),
                    round(d1["ang_brazo_der"], 2),
                    round(d1["ang_brazo_izq"], 2),
                    round(d1["ang_pierna_der"], 2),
                    round(d1["ang_pierna_izq"], 2),
                    round(mov_rel, 3)
                ])

    df_out = pd.DataFrame(resultados, columns=[
        "categoria", "video", "frame", "persona_1", "persona_2",
        "distancia_centros", "velocidad_relativa", "aceleracion_prom",
        "diferencia_orientacion", "ang_brazo_der", "ang_brazo_izq",
        "ang_pierna_der", "ang_pierna_izq", "movimiento_relativo"
    ])

    df_out.to_csv(csv_salida, index=False)
    print(f"\nCSV extendido guardado en: {csv_salida}")


# ---------------------------
# Ejecución principal
# ---------------------------
if __name__ == "__main__":
    ruta_base = "/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados"
    csv_entrada = os.path.join(ruta_base, "salida_deteccion/vectores_keypoints_full.csv")
    csv_salida = os.path.join(ruta_base, "salida_deteccion/interacciones_extendido_local.csv")

    calcular_interacciones_extendido(csv_entrada, csv_salida)
