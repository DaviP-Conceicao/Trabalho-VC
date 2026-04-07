import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# carregado imagens (Ajustar caminhos se necessario)
caminho_img = 'imagem21.jpg'
caminho_gt = '/home/paulinodavi/Downloads/Trabalho/21_manual1.gif'
caminho_gt = '21_manual1.gif'

img_colorida = cv2.imread(caminho_img)

if img_colorida is None:
    print(f"Erro imagem original: {caminho_img}")
    exit()
    

img_verde = img_colorida[:, :, 1].astype(np.uint8)
    
#Ler imagens GIF
cap = cv2.VideoCapture(caminho_gt)
ret, frame_gt = cap.read()
if not ret:
    print(f"Erro img Ground Truth: {caminho_gt}")
    exit()
    
#Convertendo o Ground Truth tons de cinza para binario 
gt_cinza = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
_, gt_binario = cv2.threshold(gt_cinza, 127, 255, cv2.THRESH_BINARY)
    
#Ajuste no 3.3 - Os vasos vão ficar brancos
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_verde)
limiar = 127

_, mascara_binaria = cv2.threshold(img_clahe, limiar, 255, cv2.THRESH_BINARY_INV)

###### Avaliação Quantitativa #######
print("Calculando Métricas...")

m = (mascara_binaria == 255)
g = (gt_binario == 255)

#Formulas 
TP = np.sum(m & g)
FP = np.sum(m & ~g)
FN = np.sum(~m & g)

# Tirar divisão por zero
Precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1_Score = (2 * Precisao * Recall) / (Precisao + Recall) if (Precisao + Recall) > 0 else 0

print(f"Precisão: {Precisao:.4f}")
print(f"Recall (Sensibilidade): {Recall:.4f}")
print(f"F1-Score: {F1_Score:.4f}")


##### Resultados######
plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_verde, cmap='gray')
plt.title('Canal verde')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(mascara_binaria, cmap='gray')
plt.title('Sua Segmentação') 
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gt_binario, cmap="gray")
plt.title('Ground Truth (Especialista)')
plt.axis('off') 


plt.figtext(0.65, 0.15, f'F1-Score: {F1_Score:.4f}', color='red', fontsize=12, weight='bold', backgroundcolor='white')

plt.tight_layout()
plt.show()