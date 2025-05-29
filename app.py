from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import os
import cv2
import base64
import numpy as np
from anpr import reconhecer_placa_robusto

app = Flask(__name__)
app.secret_key = 'placaview_super_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Cria pasta de upload se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Conexão com MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['ocr_db']
collection = db['placas']

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_base64 = request.form.get("image_base64")

        # Captura via câmera
        if image_base64:
            try:
                header, encoded = image_base64.split(",", 1)
                image_data = base64.b64decode(encoded)
                nparr = np.frombuffer(image_data, np.uint8)
                imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                texto_placa, imagem_resultado = reconhecer_placa_robusto(imagem)

                if imagem_resultado is None or imagem_resultado.size == 0:
                    flash("❌ Erro: não foi possível processar a imagem capturada.")
                    return redirect(url_for('upload_image'))

                _, buffer = cv2.imencode('.png', imagem_resultado)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                collection.insert_one({
                    'placa': texto_placa,
                    'filename': 'capturada_webcam.png',
                    'image_base64': img_base64,
                    'hora_entrada': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'hora_saida': None
                })

                return render_template('index.html', placa=texto_placa, image=img_base64)
            except Exception as e:
                flash("❌ Erro ao processar imagem da câmera.")
                return redirect(url_for('upload_image'))

        # Upload tradicional
        else:
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                texto_placa, imagem_resultado = reconhecer_placa_robusto(filepath)

                if imagem_resultado is None or imagem_resultado.size == 0:
                    flash("❌ Erro: não foi possível processar a imagem enviada.")
                    return redirect(url_for('upload_image'))

                _, buffer = cv2.imencode('.png', imagem_resultado)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                collection.insert_one({
                    'placa': texto_placa,
                    'filename': filename,
                    'image_base64': img_base64,
                    'hora_entrada': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'hora_saida': None
                })

                return render_template('index.html', placa=texto_placa, image=img_base64)
            else:
                flash("❌ Nenhuma imagem foi enviada.")
                return redirect(url_for('upload_image'))

    return render_template('index.html', placa=None)

@app.route('/registros')
def ver_registros():
    registros = list(collection.find().sort('_id', -1))
    return render_template('registros.html', registros=registros)

@app.route('/delete/<id>', methods=['POST'])
def deletar(id):
    registro = collection.find_one({'_id': ObjectId(id)})

    if registro and 'filename' in registro:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], registro['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)

    collection.delete_one({'_id': ObjectId(id)})
    flash("🗑 Registro excluído com sucesso.")
    return redirect(url_for('ver_registros'))

@app.route('/buscar', methods=['GET', 'POST'])
def buscar_placa():
    resultado = None
    mensagem = None

    if request.method == 'POST':
        placa_busca = request.form['placa'].strip().upper()
        resultado = collection.find_one({'placa': placa_busca})
        if not resultado:
            mensagem = f"A placa {placa_busca} não foi encontrada no sistema."

    return render_template('buscar.html', resultado=resultado, mensagem=mensagem)

@app.route('/editar/<id>', methods=['GET', 'POST'])
def editar(id):
    registro = collection.find_one({'_id': ObjectId(id)})

    if not registro:
        return "Registro não encontrado.", 404

    if request.method == 'POST':
        placa = request.form['placa'].strip().upper()
        entrada = request.form['hora_entrada'].strip()
        saida = request.form['hora_saida'].strip() or None

        collection.update_one(
            {'_id': ObjectId(id)},
            {'$set': {
                'placa': placa,
                'hora_entrada': entrada,
                'hora_saida': saida
            }}
        )
        flash('✅ Dados atualizados com sucesso!')
        return redirect(url_for('ver_registros'))

    return render_template('editar.html', registro=registro)

if __name__ == '__main__':
    app.run(debug=True)