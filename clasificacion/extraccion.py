import mahotas as mt

def extraer_caracteristicas( imagen ):
    return mt.features.haralick( imagen, return_mean=True )
