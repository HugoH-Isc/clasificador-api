import cv2

def preprocesar( imagen ):
    hsi = convertir_a_hsi( imagen )
    ( h, s, i ) = cv2.split( hsi )
    clahe = cv2.createCLAHE( clipLimit=0.8, tileGridSize=(8,8) )
    i_equ = clahe.apply( i )
    hsi_equ = cv2.merge( ( h, s, i_equ ) )
    return convertir_a_rgb( hsi_equ )

def convertir_a_hsi( imagen_rgb ):
    return cv2.cvtColor( imagen_rgb, cv2.COLOR_RGB2HSV )

def convertir_a_rgb( imagen_hsi ):
    return cv2.cvtColor( imagen_hsi, cv2.COLOR_HSV2RGB )
