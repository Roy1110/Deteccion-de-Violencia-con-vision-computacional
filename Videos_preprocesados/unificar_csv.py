import os
import pandas as pd
from tqdm import tqdm

# CONFIGURACIÓN DE RUTAS
ruta_temp = "/home/rodrigo/6tosemestre/Computo Paralelo/proyecto_violencia/Videos_preprocesados/salida_deteccion/temp_videos"
salida_final = os.path.join(os.path.dirname(ruta_temp), "vectores_keypoints_full.csv")

# DETECCIÓN DE ARCHIVOS CSV
archivos_csv = [os.path.join(ruta_temp, f) for f in os.listdir(ruta_temp) if f.endswith(".csv")]
print(f"Se encontraron {len(archivos_csv)} archivos temporales.\n")

if not archivos_csv:
    raise FileNotFoundError("No se encontraron archivos CSV en la carpeta temporal.")

# UNIFICACIÓN DE ARCHIVOS CON PROGRESO
dataframes = []
for archivo in tqdm(archivos_csv, desc="Unificando archivos CSV"):
    try:
        df_temp = pd.read_csv(archivo)
        # Validar columnas esperadas
        if not {"video", "frame"}.issubset(df_temp.columns):
            print(f"Archivo {os.path.basename(archivo)} omitido (faltan columnas esperadas).")
            continue
        dataframes.append(df_temp)
    except Exception as e:
        print(f"Error al leer {archivo}: {e}")

# COMBINAR TODOS LOS CSV
if not dataframes:
    raise ValueError("No se pudo combinar ningún archivo CSV válido.")

df_final = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal de filas combinadas: {len(df_final):,}")

# ELIMINAR DUPLICADOS
columnas_clave = ["categoria", "video", "frame", "persona_id"] if "persona_id" in df_final.columns else ["video", "frame"]
filas_antes = len(df_final)
df_final.drop_duplicates(subset=columnas_clave, inplace=True)
filas_despues = len(df_final)
duplicados_eliminados = filas_antes - filas_despues

print(f"Duplicados eliminados: {duplicados_eliminados:,}")
print(f"Total final de filas: {filas_despues:,}")

# ORDENAR POR VIDEO Y FRAME
if {"video", "frame"}.issubset(df_final.columns):
    df_final.sort_values(by=["video", "frame"], inplace=True)

# GUARDAR CSV FINAL
df_final.to_csv(salida_final, index=False)
print(f"\nCSV unificado guardado correctamente en:\n{salida_final}")

# VERIFICACIÓN FINAL
print("\nResumen final:")
print(f" Archivos procesados: {len(archivos_csv)}")
print(f" Filas totales: {len(df_final):,}")
print(f" Duplicados eliminados: {duplicados_eliminados:,}")
