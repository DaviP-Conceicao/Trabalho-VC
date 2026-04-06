import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CARREGAR IMAGEM E EXTRAIR CANAL VERDE
# ============================================
caminho = 'imagem21.jpg'
img_colorida = cv2.imread(caminho)
img_verde = img_colorida[:, :, 1].astype(np.uint8)

# ============================================
# APLICAR A MELHOR TRANSFORMAÇÃO (CLAHE)
# ============================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_verde)

# ============================================
# APLICAR LIMIARIZAÇÃO SIMPLES
# ============================================
# Limiar fixo
limiar = 127
_, mascara_binaria = cv2.threshold(img_clahe, limiar, 255, cv2.THRESH_BINARY)

# ============================================
# EXIBIR RESULTADOS
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_verde, cmap='gray')
plt.title('Canal Verde (Original)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mascara_binaria, cmap='gray')
plt.title(f'Máscara Binária (T={limiar})')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Limiar utilizado: {limiar}")
print("Máscara: 0 = fundo (preto) | 255 = vaso (branco)")