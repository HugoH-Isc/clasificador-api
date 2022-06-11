from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import pdfkit
from api2pdf import Api2Pdf
import numpy as np
import cv2

from clasificacion import clasificador

api = Api2Pdf('5e7ee2fc-1070-4809-9984-0be23e9f734c')
app = Flask(__name__)
CORS(app)

@app.route('/api/clasificar', methods=['POST'])
def clasificar():
    try:
        data = request.files['imagen'].read()
        img = np.frombuffer( data, np.uint8 )
        archivo = cv2.imdecode( img, cv2.IMREAD_COLOR)
        resultado = clasificador.clasificar( archivo )
        return jsonify(ok=True,clasificacion=resultado)
    except Exception as e :
        app.logger.info("Error: ", e)
        return jsonify(ok=False,error='Error al clasificar imagen')

@app.route('/api/reporte', methods=['POST'])
def generarReporte():
    try:
        data = request.files['imagen'].read()
        img = np.frombuffer( data, np.uint8 )
        archivo = cv2.imdecode( img, cv2.IMREAD_COLOR)
        template = clasificador.generar_reporte( archivo )
        response = api.Chrome.html_to_pdf( template )
        return jsonify( response.result ) 
    except Exception as e:
        app.logger.info("Error ", e)
        return jsonify(ok=False,error='Error al clasificar imagen')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

