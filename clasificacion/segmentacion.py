import cv2
import numpy as np

def segmentar( imagen, extraer_segmentada=False ):
    imagen_segmentada, centros = segmentar_imagen( imagen )
    centros_ordenados = sorted( centros, key = lambda x : ( x[0], x[1], x[2] ) )
    c_blanco = centros_ordenados[2]
    c_rosa = centros_ordenados[1]
    c_purpura = centros_ordenados[0]
    
    if( extraer_segmentada == False ):
        return extraer_cluster( c_blanco, imagen_segmentada ), extraer_cluster( c_rosa, imagen_segmentada ), extraer_cluster( c_purpura, imagen_segmentada )
    else: 
        return imagen_segmentada.reshape(imagen.shape), extraer_cluster( c_blanco, imagen_segmentada ), extraer_cluster( c_rosa, imagen_segmentada ), extraer_cluster( c_purpura, imagen_segmentada )

def segmentar_imagen( imagen ):
    pixeles = imagen.reshape(( -1, 3 ))
    pixeles = np.float32( pixeles )
    criterio_paro = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.8 )
    k = 3
    _, etiquetas, ( centros ) = cv2.kmeans( pixeles, k, None, criterio_paro, 10, cv2.KMEANS_RANDOM_CENTERS )
    centros = np.uint8( centros )
    etiquetas = etiquetas.flatten()
    return centros[etiquetas.flatten()], centros

def extraer_cluster( cluster, imagen ):
    imagen_nueva = np.zeros( imagen.shape, dtype=np.uint8 )
    j = 0
    for pixel in imagen:
        if ( pixel == cluster ).all():
            imagen_nueva[j] = [255, 255, 255]
        j+=1
    return imagen_nueva



