from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
from os import remove
import os
import cv2
import deteccion_daños

from datetime import timedelta

#Configuración de archivos permitidos.
ALLOWED_EXTENSIONS = set(['png','jpg', 'jpeg' ,'JPG','PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

#Establecer el tiempo de caducidad de la caché de archivos estáticos
app.send_file_max_age_default = timedelta(seconds=1)

nombre_cuandro_actual = ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def SubirImagen():
    return render_template("formImagen.html")

@app.route("/home/cuadro", methods = ["POST"])
def deteccion():
    if request.method == "POST":

        f = request.files['fichero']
        name_file = f.filename
        

        if not(f and allowed_file(f.filename)):
            return jsonify ({"error": 1001, "msg": "Verifique el tipo de imagen cargada, solo png, jpg, jpeg, JPG, PNG"})

        #name_file = request.form.get["nombre"]
        # Ruta del archivo actual
        basepath = os.path.dirname(__file__)

        # Como estamos en windows hay que invertir los caracteres "/"
        carpeta_imagenes = basepath + '/static/images/'
        upload_path = os.path.join(carpeta_imagenes , f.filename)
        upload_path = upload_path.replace('/',chr(92))

        f.save(upload_path)

        img = cv2.imread(upload_path)

        upload_curren = os.path.join(carpeta_imagenes ,'test.jpg' )
        cv2.imwrite(upload_curren, img)

        return render_template('deteccion_daño.html', name_file=name_file )

@app.route('/home/cuadro/daños')
def MostrarDaños():
    nombre = request.args.get('nombre')
    print(nombre)

    basepath = os.path.dirname(__file__)

    # Como estamos en windows hay que invertir los caracteres "/"
    carpeta_imagenes = basepath + '/static/images/'
    carpeta_imagenes = carpeta_imagenes.replace('/',chr(92))


    # Una vez que tenemos la imagen detectamos los daños
    deteccion_daños.guardarImagenDaños(nombre)



    """
        f = request.files['fichero']

        if not(f and allowed_file(f.filename)):
            return jsonify ({"error": 1001, "msg": "Verifique el tipo de imagen cargada, solo png, jpg, jpeg, JPG, PNG"})

        #name_file = request.form.get["nombre"]
        # Ruta del archivo actual
        basepath = os.path.dirname(__file__)

        # Como estamos en windows hay que invertir los caracteres "/"
        carpeta_imagenes = basepath + '/static/images/'
        upload_path = os.path.join(carpeta_imagenes , f.filename)
        upload_path = upload_path.replace('/',chr(92))

        nombre_imagen = f.filename

        f.save(upload_path)

        # Una vez que tenemos la imagen detectamos los daños
        deteccion_daños.guardarImagenDaños(nombre_imagen)
    """

    return render_template('mostrar_daños.html')

@app.route('/home/cuadro/daños/download')
def DownloadImages():
  pathfile = '/content/static/images/'
  path='deteccion.jpg'
  return send_from_directory(pathfile,path,as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
