from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import cv2
import base64
from anpr import reconhecer_placa_robusto  # seu código está em anpr.py

# Configuração da pasta de uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicialização do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Conexão com o MongoDB local (ajuste se estiver usando MongoDB Atlas)
client = MongoClient('mongodb://localhost:27017/')
db = client['ocr_db']
collection = db['placas']

# Página inicial (upload)
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            texto_placa, imagem_resultado = reconhecer_placa_robusto(filepath)

            # Converte imagem com bbox para base64
            _, buffer = cv2.imencode('.png', imagem_resultado)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Salva no banco
            collection.insert_one({
                'placa': texto_placa,
                'filename': filename,
                'image_base64': img_base64
            })

            return render_template('index.html', placa=texto_placa, image=img_base64)

    return render_template('index.html', placa=None)

# Página de registros salvos
@app.route('/registros')
def ver_registros():
    registros = list(collection.find().sort('_id', -1))
    return render_template('registros.html', registros=registros)

# Função para deletar um registro
@app.route('/delete/<id>', methods=['POST'])
def deletar(id):
    registro = collection.find_one({'_id': ObjectId(id)})
    
    # Exclui arquivo da pasta uploads (se existir)
    if registro and 'filename' in registro:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], registro['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)

    # Remove do banco de dados
    collection.delete_one({'_id': ObjectId(id)})

    return redirect(url_for('ver_registros'))

# Rota de teste opcional
@app.route('/teste')
def teste():
    return "Rota de teste funcionando!"

# Inicia o servidor
if __name__ == '__main__':
    app.run(debug=True)
