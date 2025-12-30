import os
import cv2
from collections import Counter

def encontrar_dimension_por_carpeta(carpetas):
    """
    Analiza archivos de video en una lista de carpetas, encuentra la dimensión
    más frecuente en general y también desglosa los resultados por carpeta.
    """
    # Usamos un diccionario para guardar las dimensiones por cada carpeta
    dimensiones_por_carpeta = {carpeta: [] for carpeta in carpetas}
    dimensiones_totales = []
    
    # Lista de extensiones de video comunes para filtrar los archivos
    extensiones_video = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']

    print("Iniciando análisis de videos")
    print("-" * 40)

    for carpeta in carpetas:
        print(f"Procesando carpeta: '{carpeta}'")
        if not os.path.isdir(carpeta):
            print(f"Advertencia: La carpeta '{carpeta}' no existe. Omitiendo.")
            continue

        # Itera sobre cada archivo en el directorio
        for nombre_archivo in os.listdir(carpeta):
            if not any(nombre_archivo.lower().endswith(ext) for ext in extensiones_video):
                continue

            ruta_video = os.path.join(carpeta, nombre_archivo)

            try:
                cap = cv2.VideoCapture(ruta_video)
                if not cap.isOpened():
                    print(f"Error: No se pudo abrir el video '{nombre_archivo}'.")
                    continue

                ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if ancho > 0 and alto > 0:
                    dimension = (ancho, alto)
                    # Agrega la dimensión a la lista específica de la carpeta y a la lista total
                    dimensiones_por_carpeta[carpeta].append(dimension)
                    dimensiones_totales.append(dimension)

                cap.release()

            except Exception as e:
                print(f"Error procesando el archivo '{nombre_archivo}': {e}")
    
    print("\nResultados del Análisis")
    
    # 1. Imprimir resultados por cada carpeta
    print("\n" + "=" * 15 + " Desglose por Carpeta " + "=" * 15)
    for carpeta, dimensiones in dimensiones_por_carpeta.items():
        print(f"\nResultados para la carpeta: '{carpeta}'")
        if not dimensiones:
            print("  -> No se encontraron videos válidos en esta carpeta.")
            continue
            
        conteo_carpeta = Counter(dimensiones)
        dim_comun_carpeta, frec_carpeta = conteo_carpeta.most_common(1)[0]

        print(f"Dimensión más frecuente: **{dim_comun_carpeta[0]}x{dim_comun_carpeta[1]}** ({frec_carpeta} veces)")
        print("Conteo de todas las dimensiones:")
        for dimension, num in conteo_carpeta.most_common():
            print(f"     - {dimension[0]}x{dimension[1]}: {num} videos")

    # 2. Imprimir el resumen general
    print("\n" + "=" * 17 + " Resumen General " + "=" * 18)
    if not dimensiones_totales:
        print("\nNo se encontraron videos válidos en ninguna de las carpetas.")
        return

    conteo_total = Counter(dimensiones_totales)
    dimension_mas_comun, frecuencia_total = conteo_total.most_common(1)[0]
    
    print(f"\nLa dimensión más frecuente en **todas** las carpetas es: **{dimension_mas_comun[0]}x{dimension_mas_comun[1]}**")
    print(f"Aparece un total de **{frecuencia_total}** veces.")
    
if __name__ == "__main__":
    carpetas_a_analizar = ["noViolencia", "violencia_limpios"]
    encontrar_dimension_por_carpeta(carpetas_a_analizar)