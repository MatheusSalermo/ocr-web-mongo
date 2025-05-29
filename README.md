# 📸 PlacaView – Estacionamento com OCR e MongoDB

**PlacaView** é um sistema web que realiza o reconhecimento de placas de veículos a partir de imagens usando OCR (EasyOCR) e registra entradas e saídas em um banco de dados MongoDB.

Ideal para controle automatizado de estacionamentos, com histórico, consulta e CRUD completo dos veículos.

---

## 🚀 Funcionalidades

- ✅ Upload de imagem com detecção automática de placa
- ✅ Registro da hora de entrada
- ✅ Armazenamento em MongoDB (imagem + dados)
- ✅ Histórico de veículos
- ✅ Exclusão de registros
- 🔍 Consulta por placa (em construção)
- ✏️ Edição (hora de saída etc.) (em construção)

---

## 🛠 Tecnologias utilizadas

- Python + Flask
- EasyOCR + OpenCV
- MongoDB (local ou Atlas)
- HTML + CSS (interface simples e responsiva)

---

## 📦 Instalação e Execução

### 1. Clone o projeto

```bash
git clone https://github.com/seu-usuario/placaview.git
cd placaview
```

### 2. Crie e ative o ambiente virtual (opcional)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Inicie o MongoDB
```bash
mongod
```

### 5. Rode o servidor Flask
```bash
python app.py
```
Acesse em: http://127.0.0.1:5000

📸 Exemplo de uso
Acesse a home /

Envie uma imagem com uma placa

A placa será reconhecida e salva com a hora de entrada

Acesse /registros para visualizar todos os veículos registrados

🧪 Em desenvolvimento
Consulta por placa

Registro de hora de saída

Edição completa de registros

Relatórios e exportações

Desenvolvido por Matheus Salermo e Bernardo Maia


