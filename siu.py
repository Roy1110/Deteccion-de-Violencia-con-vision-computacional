import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import glob
import torch

def obtener_cfg():
    """Función para obtener la configuración del modelo forzando CPU"""
    cfg = get_cfg()
    # Forzar uso de CPU
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def predecir_keypoints(ruta_imagen):
    """
    Realiza la detección de puntos clave en una imagen usando CPU
    
    Args:
        ruta_imagen (str): Ruta a la imagen a procesar
    """
    try:
        # Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen desde {ruta_imagen}")
            return None
        
        print(f"Procesando: {ruta_imagen} - Tamaño: {imagen.shape}")
        
        # Configuración del modelo para CPU
        cfg_keypoint = obtener_cfg()
        cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        
        # Crear predictor en CPU
        predictor = DefaultPredictor(cfg_keypoint)
        
        # Realizar predicción
        outputs = predictor(imagen)
        
        # Visualizar resultados
        v = Visualizer(imagen[:, :, ::-1], 
                       MetadataCatalog.get(cfg_keypoint.DATASETS.TRAIN[0]), 
                       scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Mostrar resultados
        plt.figure(figsize=(12, 8))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.title(f'Detección de Puntos Clave - {os.path.basename(ruta_imagen)}')
        plt.tight_layout()
        plt.show()
        
        # Información adicional
        instances = outputs["instances"]
        print(f"Personas detectadas: {len(instances)}")
        if len(instances) > 0:
            print(f"Puntos clave detectados: {instances.pred_keypoints.shape}")
        
        return outputs
        
    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {e}")
        return None

def procesar_directorio_imagenes(directorio, max_imagenes=None):
    """
    Procesa todas las imágenes en un directorio
    
    Args:
        directorio (str): Ruta al directorio con imágenes
        max_imagenes (int): Límite máximo de imágenes a procesar (opcional)
    """
    # Extensiones de imagen soportadas
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    rutas_imagenes = []
    for extension in extensiones:
        rutas_imagenes.extend(glob.glob(os.path.join(directorio, extension)))
        rutas_imagenes.extend(glob.glob(os.path.join(directorio, extension.upper())))
    
    # Limitar número de imágenes si se especifica
    if max_imagenes:
        rutas_imagenes = rutas_imagenes[:max_imagenes]
    
    print(f"Encontradas {len(rutas_imagenes)} imágenes para procesar")
    print(f"Usando dispositivo: CPU")
    
    for i, ruta in enumerate(rutas_imagenes):
        print(f"\n--- Procesando imagen {i+1}/{len(rutas_imagenes)} ---")
        predecir_keypoints(ruta)

def guardar_resultados(ruta_imagen, outputs, directorio_salida="resultados"):
    """
    Guarda los resultados en archivos en lugar de mostrarlos
    
    Args:
        ruta_imagen (str): Ruta a la imagen original
        outputs: Resultados del predictor
        directorio_salida (str): Directorio donde guardar los resultados
    """
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    
    # Cargar imagen original
    imagen = cv2.imread(ruta_imagen)
    
    # Configuración para visualización
    cfg_keypoint = obtener_cfg()
    cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    
    # Visualizar
    v = Visualizer(imagen[:, :, ::-1], 
                   MetadataCatalog.get(cfg_keypoint.DATASETS.TRAIN[0]), 
                   scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Guardar imagen con detecciones
    nombre_archivo = os.path.basename(ruta_imagen)
    ruta_salida = os.path.join(directorio_salida, f"resultado_{nombre_archivo}")
    cv2.imwrite(ruta_salida, out.get_image()[:, :, ::-1])
    print(f"Resultado guardado en: {ruta_salida}")

# Versión optimizada para procesamiento por lotes (sin mostrar imágenes)
def procesar_lote_sin_visualizacion(directorio, directorio_salida="resultados_keypoints"):
    """
    Procesa todas las imágenes y guarda los resultados sin mostrar en pantalla
    """
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    
    # Configuración del modelo (una sola vez)
    cfg_keypoint = obtener_cfg()
    cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg_keypoint)
    
    # Encontrar imágenes
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    rutas_imagenes = []
    for extension in extensiones:
        rutas_imagenes.extend(glob.glob(os.path.join(directorio, extension)))
    
    print(f"Procesando {len(rutas_imagenes)} imágenes en modo lote...")
    
    for i, ruta in enumerate(rutas_imagenes):
        try:
            print(f"Procesando {i+1}/{len(rutas_imagenes)}: {os.path.basename(ruta)}")
            
            # Cargar y procesar imagen
            imagen = cv2.imread(ruta)
            outputs = predictor(imagen)
            
            # Guardar resultados
            guardar_resultados(ruta, outputs, directorio_salida)
            
            # Información de progreso
            instances = outputs["instances"]
            print(f"  → Personas detectadas: {len(instances)}")
            
        except Exception as e:
            print(f"  → Error: {e}")

if __name__ == "__main__":
    # OPCIÓN 1: Procesar con visualización (máximo 5 imágenes para prueba)
    # procesar_directorio_imagenes(
    #     "/home/rodrigo/6tosemestre/metodologia/proyecto/Deteccion-de-violencia/Videos_preprocesados/imagenes_keypoints/violencia/V_228",
    #     max_imagenes=5
    # )
    
    # OPCIÓN 2: Procesar en lote y guardar resultados (recomendado para muchas imágenes)
    procesar_lote_sin_visualizacion(
        "/home/rodrigo/6tosemestre/metodologia/proyecto/Deteccion-de-violencia/Videos_preprocesados/imagenes_keypoints/violencia/V_228",
        directorio_salida="resultados_keypoints"
    )