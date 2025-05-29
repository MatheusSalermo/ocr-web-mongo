# -*- coding: utf-8 -*-
"""
Sistema AVANÇADO de OCR para reconhecimento de placas de veículos usando OpenCV e EasyOCR.
Versão com detecção robusta de placas para casos reais.
"""

import cv2
import easyocr
import numpy as np
import sys
import os
import re

# Inicializa o leitor EasyOCR
try:
    # Adiciona português e inglês
    reader = easyocr.Reader(["pt", "en"], gpu=False) 
except Exception as e:
    print(f"Erro ao inicializar EasyOCR: {e}")
    reader = None

def corrigir_caracteres_similares(texto):
    """
    Corrige caracteres frequentemente confundidos pelo OCR em placas.
    
    Args:
        texto: Texto da placa reconhecida
        
    Returns:
        str: Texto corrigido
    """
    if not texto:
        return texto
        
    # Mapeamento de correções comuns em placas brasileiras
    # Baseado em padrões conhecidos e confusões frequentes do OCR
    correcoes = {
        # Letras confundidas com números
        'O': '0',  # Letra O -> Número 0 (comum em placas)
        'I': '1',  # Letra I -> Número 1
        'Z': '2',  # Letra Z -> Número 2
        'S': '5',  # Letra S -> Número 5
        'G': '6',  # Letra G -> Número 6
        'B': '8',  # Letra B -> Número 8
        
        # Números confundidos com letras (em posições específicas)
        # Estas correções são aplicadas apenas em posições específicas
    }
    
    # Aplica correções gerais
    texto_corrigido = ""
    for i, char in enumerate(texto):
        # Verifica o padrão da placa para aplicar correções específicas por posição
        if len(texto) == 7:  # Placas brasileiras têm 7 caracteres
            # Padrão Mercosul: 3 letras + 1 número + 1 letra + 2 números
            if i < 3:  # Primeiras 3 posições são letras
                # Se for um número em posição de letra, converte para letra
                if char.isdigit():
                    # Mapeamento inverso para posições de letras
                    inv_map = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
                    char = inv_map.get(char, char)
            elif i == 3 or i >= 5:  # Posições de números (4ª, 6ª e 7ª)
                # Se for uma letra em posição de número, converte para número
                if char.isalpha():
                    char = correcoes.get(char, char)
            # A 5ª posição (índice 4) pode ser letra no padrão Mercosul
        
        texto_corrigido += char
    
    return texto_corrigido

def filtrar_texto_placa(texto_bruto):
    """
    Filtra o texto bruto reconhecido para extrair a sequência da placa.
    Prioriza padrões conhecidos (AAA1B23, AAA1234), mas aceita qualquer sequência de 7.
    """
    if not texto_bruto:
        return None
        
    texto_limpo = "".join(texto_bruto.split()).upper()
    
    # Padrão Mercosul (AAA1B23)
    match_mercosul = re.search(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', texto_limpo)
    if match_mercosul:
        return corrigir_caracteres_similares(match_mercosul.group(0))
    
    # Padrão antigo (AAA1234) - pode vir com ou sem hífen
    texto_sem_hifen = texto_limpo.replace('-', '')
    match_antigo = re.search(r'[A-Z]{3}[0-9]{4}', texto_sem_hifen)
    if match_antigo:
        return corrigir_caracteres_similares(match_antigo.group(0))
    
    # Padrão antigo com hífen explícito (AAA-1234)
    match_antigo_hifen = re.search(r'[A-Z]{3}-[0-9]{4}', texto_limpo)
    if match_antigo_hifen:
        return match_antigo_hifen.group(0)  # Mantém o hífen se detectado explicitamente
        
    # Fallback genérico (menos confiável)
    match_generico = re.search(r'[A-Z0-9]{7}', texto_limpo)
    if match_generico and len(texto_limpo) < 15:
        return corrigir_caracteres_similares(match_generico.group(0))

    return None

def detectar_placa_por_texto_direto(imagem):
    """
    Abordagem robusta: detecta a placa procurando diretamente por texto em toda a imagem
    e filtrando por padrões de placa.
    
    Args:
        imagem: Imagem de entrada (numpy array)
        
    Returns:
        tuple: (texto_placa, bbox, imagem_com_bbox)
    """
    if reader is None:
        return None, None, imagem
    
    # Cria uma cópia da imagem para desenhar o bbox
    imagem_com_bbox = imagem.copy()
    
    # Executa OCR em toda a imagem
    resultados = reader.readtext(imagem)
    
    # Filtra resultados que parecem placas
    candidatos_placa = []
    for (bbox, texto, confianca) in resultados:
        # Verifica se o texto parece uma placa
        texto_filtrado = filtrar_texto_placa(texto)
        if texto_filtrado:
            # Calcula a área do bbox
            (tl, tr, br, bl) = bbox
            largura = max(np.linalg.norm(np.array(tr) - np.array(tl)), 
                         np.linalg.norm(np.array(br) - np.array(bl)))
            altura = max(np.linalg.norm(np.array(bl) - np.array(tl)), 
                        np.linalg.norm(np.array(br) - np.array(tr)))
            area = largura * altura
            
            # Armazena o candidato com sua área e confiança
            candidatos_placa.append((texto_filtrado, bbox, confianca, area))
    
    # Se não encontrou candidatos, retorna None
    if not candidatos_placa:
        return None, None, imagem
    
    # Ordena por confiança e área (priorizando textos maiores e mais confiáveis)
    candidatos_placa.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    # Pega o melhor candidato
    melhor_texto, melhor_bbox, _, _ = candidatos_placa[0]
    
    # Desenha o bbox na imagem
    pts = np.array(melhor_bbox, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(imagem_com_bbox, [pts], True, (0, 255, 0), 3)
    
    return melhor_texto, melhor_bbox, imagem_com_bbox

def detectar_placa_por_segmentacao_cor(imagem):
    """
    Detecta a placa usando segmentação por cor (focando em regiões cinza/prata).
    
    Args:
        imagem: Imagem de entrada (numpy array)
        
    Returns:
        tuple: (região_placa, imagem_com_bbox)
    """
    # Cria uma cópia da imagem para desenhar o bbox
    imagem_com_bbox = imagem.copy()
    
    # Converte para HSV para melhor segmentação de cor
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Define faixas de cor para cinza/prata (placas brasileiras)
    # Estes valores podem precisar de ajuste
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 220])
    
    # Cria máscara para regiões cinza/prata
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Aplica operações morfológicas para limpar a máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Encontra contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra contornos por área e proporção (placas têm proporção específica)
    candidatos_placa = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        proporcao = w / float(h)
        
        # Placas brasileiras têm proporção aproximada de 3:1
        if area > 1000 and 2.0 < proporcao < 5.0:
            candidatos_placa.append((x, y, w, h, area))
    
    # Se não encontrou candidatos, retorna None
    if not candidatos_placa:
        return None, imagem_com_bbox
    
    # Ordena por área (maior primeiro)
    candidatos_placa.sort(key=lambda x: x[4], reverse=True)
    
    # Pega o melhor candidato
    x, y, w, h, _ = candidatos_placa[0]
    
    # Recorta a região da placa
    regiao_placa = imagem[y:y+h, x:x+w]
    
    # Desenha o bbox na imagem
    cv2.rectangle(imagem_com_bbox, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    return regiao_placa, imagem_com_bbox

def reconhecer_placa_robusto(caminho_imagem):
    """
    Pipeline robusto para reconhecimento de placas em imagens reais.
    Combina múltiplas abordagens de detecção.
    
    Args:
        caminho_imagem: Caminho para a imagem
        
    Returns:
        tuple: (texto_placa, imagem_resultado)
    """
    if reader is None:
        print("Erro: Leitor EasyOCR não inicializado.")
        return None, None
    
    try:
        # Carrega a imagem
        if not os.path.exists(caminho_imagem):
            print(f"Erro: Arquivo não encontrado: '{caminho_imagem}'")
            return None, None
            
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem: '{caminho_imagem}'")
            return None, None
        
        # ABORDAGEM 1: Detecção direta de texto com padrão de placa
        print("Tentando detecção por texto direto...")
        texto_placa, bbox_texto, imagem_resultado_texto = detectar_placa_por_texto_direto(imagem)
        
        if texto_placa:
            print(f"Placa detectada por texto direto: {texto_placa}")
            
            # Formata a placa de acordo com o padrão (adiciona hífen se for placa antiga)
            if len(texto_placa) == 7 and texto_placa[3].isdigit() and texto_placa[0:3].isalpha() and texto_placa[4:].isdigit():
                # Parece ser placa antiga (AAA1234), adiciona hífen
                texto_placa_formatado = f"{texto_placa[0:3]}-{texto_placa[3:]}"
                print(f"Placa formatada: {texto_placa_formatado}")
                return texto_placa_formatado, imagem_resultado_texto
            
            return texto_placa, imagem_resultado_texto
        
        # ABORDAGEM 2: Segmentação por cor
        print("Tentando detecção por segmentação de cor...")
        regiao_placa, imagem_resultado_cor = detectar_placa_por_segmentacao_cor(imagem)
        
        if regiao_placa is not None:
            # Aplica OCR na região da placa
            resultados_ocr = reader.readtext(regiao_placa)
            
            # Concatena todos os textos encontrados
            texto_bruto = " ".join([texto for _, texto, _ in resultados_ocr])
            print(f"Texto bruto da região segmentada: {texto_bruto}")
            
            # Filtra o texto para extrair a placa
            texto_placa = filtrar_texto_placa(texto_bruto)
            
            if texto_placa:
                print(f"Placa detectada por segmentação de cor: {texto_placa}")
                
                # Formata a placa de acordo com o padrão (adiciona hífen se for placa antiga)
                if len(texto_placa) == 7 and texto_placa[3].isdigit() and texto_placa[0:3].isalpha() and texto_placa[4:].isdigit():
                    # Parece ser placa antiga (AAA1234), adiciona hífen
                    texto_placa_formatado = f"{texto_placa[0:3]}-{texto_placa[3:]}"
                    print(f"Placa formatada: {texto_placa_formatado}")
                    return texto_placa_formatado, imagem_resultado_cor
                
                return texto_placa, imagem_resultado_cor
        
        # Se nenhuma abordagem funcionou, retorna None
        print("Nenhuma placa detectada com as abordagens tentadas.")
        return None, imagem
        
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None, imagem if 'imagem' in locals() else None

# Exemplo de uso
if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho_arquivo_imagem = sys.argv[1]
    else:
        # Tenta usar a imagem fornecida pelo usuário
        caminho_arquivo_imagem = '/home/ubuntu/upload/image.png'
        if not os.path.exists(caminho_arquivo_imagem):
            print(f"Aviso: Imagem de teste não encontrada em '{caminho_arquivo_imagem}'.")
            print("Por favor, forneça um caminho de imagem como argumento.")
            sys.exit(1)
        else:
            print(f"Usando imagem: '{caminho_arquivo_imagem}'")
    
    texto_reconhecido, imagem_resultado = reconhecer_placa_robusto(caminho_arquivo_imagem)
    
    if texto_reconhecido:
        print(f"\n--- Placa Reconhecida (Método Robusto) ---")
        print(f"Texto Final: {texto_reconhecido}")
        if imagem_resultado is not None:
            output_path = "/home/ubuntu/placa_detectada_robusta.png"
            cv2.imwrite(output_path, imagem_resultado)
            print(f"Imagem com detecção salva em: {output_path}")
    else:
        print("\n--- Não foi possível reconhecer um padrão de placa válido. ---")
        if imagem_resultado is not None:
            output_path = "/home/ubuntu/placa_nao_reconhecida_robusta.png"
            cv2.imwrite(output_path, imagem_resultado)
            print(f"Imagem com tentativa de detecção salva em: {output_path}")
