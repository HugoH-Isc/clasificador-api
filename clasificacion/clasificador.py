import io
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
from flask import render_template 

from clasificacion import preprocesamiento
from clasificacion import segmentacion
from clasificacion import extraccion
from clasificacion import modelo

cabeceras = ['u-b','con-b','corr-b','v-b','h-b','sp-b','vs-b','es-b','e-b','vd-b','ed-b','imc1-b','imc2-b',
             'u-r','con-r','corr-r','v-r','h-r','sp-r','vs-r','es-r','e-r','vd-r','ed-r','imc1-r','imc2-r',
             'u-p','con-p','corr-p','v-p','h-p','sp-p','vs-p','es-p','e-p','vd-p','ed-p','imc1-p','imc2-p']

componentes = ['PC1', 'PC2', 'PC3','PC4','PC5']

def clasificar( imagen ):
    
    imagen_preprocesada = preprocesamiento.preprocesar( imagen )
    
    cluster_blanco, cluster_rosa, cluster_purpura = segmentacion.segmentar( imagen_preprocesada )
    
    caracteristicas_b = np.array(extraccion.extraer_caracteristicas( cluster_blanco ))
    caracteristicas_r = np.array(extraccion.extraer_caracteristicas( cluster_rosa ))
    caracteristicas_p = np.array(extraccion.extraer_caracteristicas( cluster_purpura ))
    
    caracteristicas = np.concatenate(( caracteristicas_b, caracteristicas_r, caracteristicas_p ))
    
    df = pd.DataFrame( data=[caracteristicas], columns=cabeceras )
    return modelo.predecir_valor( df )[0].item()

def generar_reporte( imagen ):
 
    imagen_preprocesada = preprocesamiento.preprocesar( imagen )
    segmentada, cluster_blanco, cluster_rosa, cluster_purpura = segmentacion.segmentar( imagen_preprocesada, extraer_segmentada=True )
    caracteristicas_b = extraccion.extraer_caracteristicas( cluster_blanco )
    caracteristicas_r = extraccion.extraer_caracteristicas( cluster_rosa )
    caracteristicas_p = extraccion.extraer_caracteristicas( cluster_purpura )
    
    caracteristicas = np.concatenate(( caracteristicas_b, caracteristicas_r, caracteristicas_p ))
    
    df = pd.DataFrame( data=[caracteristicas], columns=cabeceras )
    
    proyecciones, resultado = modelo.predecir_valor( df, RETORNAR_COMPONENTES=True)
        
    #resultado = resultado[0].item()
    
    original = crear_imagen_reporte( imagen )
    preprocesada = crear_imagen_reporte( imagen_preprocesada )
    segmentada_rep = crear_imagen_reporte( segmentada )
    segmentada_b = crear_imagen_reporte( cluster_blanco.reshape( imagen.shape ) ) 
    segmentada_r = crear_imagen_reporte( cluster_rosa.reshape( imagen.shape ) ) 
    segmentada_p = crear_imagen_reporte( cluster_purpura.reshape( imagen.shape ) )

    return render_template('reporte.html',
                           original=original,
                           preprocesada=preprocesada,
                           segmentada=segmentada_rep,
                           c_blanco=segmentada_b,
                           c_rosa=segmentada_r,
                           c_purpura=segmentada_p,
                           resultado=resultado,
                           cabeceras=cabeceras,
                           caracteristicas=caracteristicas,
                           componentes=['pc1','pc2','pc3','pc4','pc5'],
                           proyecciones=proyecciones
                          )

def crear_imagen_reporte( imagen ):
    pil_img = Image.fromarray( imagen )
    salida = io.BytesIO()
    pil_img.save( salida, "JPEG" )
    imagen_codificada = base64.b64encode( salida.getvalue() )
    return imagen_codificada.decode('utf-8')

