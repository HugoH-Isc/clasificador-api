import joblib

def predecir_valor( caracteristicas, RETORNAR_COMPONENTES = False ):
    modelo = joblib.load("./clasificacion/knn-5.pkl")
    if( RETORNAR_COMPONENTES ):
        scaler = modelo['scaler']
        pca = modelo['pca']
        proyecciones = pca.transform(scaler.transform(caracteristicas))
        return proyecciones[0], modelo.predict( caracteristicas )[0].item()
    return modelo.predict( caracteristicas )
