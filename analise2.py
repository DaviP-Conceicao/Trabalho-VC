import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. CARREGAR IMAGEM E EXTRAIR CANAL VERDE
# ============================================
caminho = 'imagem21.jpg'  # ajuste o nome se necessário
img_colorida = cv2.imread(caminho)

if img_colorida is None:
    print("Erro: imagem não encontrada")
    exit()

img_verde = img_colorida[:, :, 1].astype(np.float32)

# ============================================
# FUNÇÃO AUXILIAR PARA PLOTAR RESULTADOS
# ============================================
def plot_resultado(img_original, img_transformada, titulo, hist_cor='blue'):
    img_orig_uint8 = np.uint8(np.clip(img_original, 0, 255))
    img_trans_uint8 = np.uint8(np.clip(img_transformada, 0, 255))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(titulo, fontsize=14)
    
    axes[0,0].imshow(img_orig_uint8, cmap='gray')
    axes[0,0].set_title('Original (Canal Verde)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img_trans_uint8, cmap='gray')
    axes[0,1].set_title(f'Transformada: {titulo}')
    axes[0,1].axis('off')
    
    hist_orig = cv2.calcHist([img_orig_uint8], [0], None, [256], [0, 256])
    axes[1,0].plot(hist_orig, color='gray')
    axes[1,0].set_title('Histograma Original')
    axes[1,0].set_xlim([0, 256])
    
    hist_trans = cv2.calcHist([img_trans_uint8], [0], None, [256], [0, 256])
    axes[1,1].plot(hist_trans, color=hist_cor)
    axes[1,1].set_title(f'Histograma Transformado')
    axes[1,1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

# ============================================
# CENÁRIO 1: TRANSFORMAÇÃO LOGARÍTMICA
# ============================================
print("Executando Cenário 1: Transformação Logarítmica...")

r_max = np.max(img_verde)
c = 255 / np.log1p(r_max)
img_log = c * np.log1p(img_verde)

plot_resultado(img_verde, img_log, 'Logarítmica', 'green')

# ============================================
# CENÁRIO 2: TRANSFORMAÇÃO DE POTÊNCIA (GAMA)
# ============================================
print("Executando Cenário 2: Transformação Gama...")

# Normalizar para [0,1] antes da transformação
img_norm = img_verde / 255.0

gammas = [0.5, 1.2, 2.0]
cores = ['red', 'orange', 'darkred']

for gamma, cor in zip(gammas, cores):
    img_gamma = c * (img_norm ** gamma)  # c=1 aqui, pois já normalizado
    img_gamma = img_gamma * 255.0
    plot_resultado(img_verde, img_gamma, f'Gama γ={gamma}', cor)

# ============================================
# CENÁRIO 3: ALARGAMENTO DE CONTRASTE
# ============================================
print("Executando Cenário 3: Alargamento de Contraste...")

r_min = np.min(img_verde)
r_max = np.max(img_verde)

img_contrast = (img_verde - r_min) / (r_max - r_min) * 255.0

plot_resultado(img_verde, img_contrast, 'Alargamento Contraste', 'purple')

# ============================================
# CENÁRIO 4: EQUALIZAÇÃO DE HISTOGRAMA
# ============================================
print("Executando Cenário 4: Equalização...")

img_verde_uint8 = np.uint8(np.clip(img_verde, 0, 255))

# 4a: Equalização Global
img_eq_global = cv2.equalizeHist(img_verde_uint8)
plot_resultado(img_verde_uint8, img_eq_global, 'Equalização Global', 'blue')

# 4b: CLAHE (Equalização Adaptativa Limitada por Contraste)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_verde_uint8)
plot_resultado(img_verde_uint8, img_clahe, 'CLAHE', 'teal')

print("Concluído! Todas as transformações foram executadas.")