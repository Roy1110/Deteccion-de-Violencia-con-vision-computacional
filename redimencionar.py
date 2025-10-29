import os
import subprocess

def procesar_video_directo(ruta_video_entrada, ruta_video_salida, ancho_out, alto_out, fps_out):
    """
    Usa subprocess para ejecutar FFmpeg directamente, evitando problemas de escapado
    de la librería ffmpeg-python.
    """
    try:
        print(f"  Procesando: {os.path.basename(ruta_video_entrada)}...")

        # Construimos la cadena de filtros exactamente como la necesita ffmpeg
        filter_string = (
            f"scale={ancho_out}:{alto_out}:force_original_aspect_ratio=decrease,"
            f"pad={ancho_out}:{alto_out}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps_out}"
        )

        # Creamos el comando como una lista de argumentos
        command = [
            'ffmpeg',
            '-y',  # Sobrescribir el archivo de salida si ya existe
            '-i', ruta_video_entrada,
            '-vf', filter_string,   # Aplicar el filtro de video
            '-c:a', 'copy',         # Copiar el audio sin recodificar
            '-loglevel', 'error',   # Solo mostrar errores en la consola
            ruta_video_salida
        ]
        
        # Ejecutamos el comando directamente
        subprocess.run(command, check=True, capture_output=True)
        
        print(f"  -> ✅ Guardado en: {ruta_video_salida} ({ancho_out}x{alto_out} @ {fps_out}fps)")

    except subprocess.CalledProcessError as e:
        # Si FFmpeg devuelve un error, lo capturamos aquí
        print(f"  -> ❌ FALLÓ (Error de FFmpeg): {os.path.basename(ruta_video_entrada)}")
        # `e.stderr` contiene el mensaje de error exacto de ffmpeg
        print(f"     Detalle del error: {e.stderr.decode('utf8').strip()}")
        
    except FileNotFoundError:
        # Este error ocurre si el comando 'ffmpeg' no se encuentra en el sistema
        print("  -> ❌ ERROR CRÍTICO: No se encontró el comando 'ffmpeg'.")
        print("     Asegúrate de que FFmpeg esté instalado y accesible desde la terminal.")
        return False # Devuelve False para detener el proceso
        
    except Exception as e:
        print(f"  -> ❌ Ocurrió un error inesperado con {os.path.basename(ruta_video_entrada)}: {e}")
        
    return True

if __name__ == "__main__":
    carpetas_origen = ["noViolencia", "violencia_limpios"]
    
    ANCHO_FINAL = 640
    ALTO_FINAL = 360
    FPS_FINAL = 30

    print(f"🚀 Iniciando conversión a {ANCHO_FINAL}x{ALTO_FINAL} y {FPS_FINAL} FPS...")
    print("-" * 50)

    for carpeta in carpetas_origen:
        carpeta_salida = f"{carpeta}_procesado"
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
            print(f"📁 Creada carpeta de salida: '{carpeta_salida}'")
        
        print(f"\nProcesando carpeta: '{carpeta}'")
        
        for nombre_archivo in os.listdir(carpeta):
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            if os.path.isfile(ruta_completa) and os.path.getsize(ruta_completa) > 0:
                ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
                if not procesar_video_directo(ruta_completa, ruta_salida, ANCHO_FINAL, ALTO_FINAL, FPS_FINAL):
                    # Si la función detecta que ffmpeg no está, detiene todo el proceso.
                    exit()
            elif os.path.getsize(ruta_completa) == 0:
                 print(f"  -> 🟡 Omitiendo archivo vacío: {nombre_archivo}")

    print("\n🎉 ¡Proceso completado!")