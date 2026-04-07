import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. CARREGAR IMAGENS
# ============================================
img_colorida = cv2.imread('imagem21.jpg')
img_verde = img_colorida[:, :, 1].astype(np.uint8)

gt = cv2.imread('21_manual1.gif', cv2.IMREAD_GRAYSCALE)
if gt is None:
    print("Erro: Ground Truth não encontrada.")
    exit()

# ============================================
# 2. GARANTIR MESMAS DIMENSÕES
# ============================================
# Se GT e imagem tiverem tamanhos diferentes, o cálculo de TP/FP/FN estará errado
if gt.shape != img_verde.shape:
    gt = cv2.resize(gt, (img_verde.shape[1], img_verde.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

# ============================================
# 3. APLICAR CLAHE + LIMIARIZAÇÃO
# ============================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_verde)

limiar = 100

# CORREÇÃO: THRESH_BINARY_INV — vasos escuros → 1, fundo claro → 0
_, mascara = cv2.threshold(img_clahe, limiar, 1, cv2.THRESH_BINARY_INV)

# ============================================
# 4. BINARIZAR GROUND TRUTH (0 e 1)
# ============================================
_, gt_binaria = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

# ============================================
# 5. CALCULAR TP, FP, FN
# ============================================
TP = np.sum((mascara == 1) & (gt_binaria == 1))  # previsto vaso, é vaso
FP = np.sum((mascara == 1) & (gt_binaria == 0))  # previsto vaso, é fundo
FN = np.sum((mascara == 0) & (gt_binaria == 1))  # previsto fundo, é vaso

# ============================================
# 6. CALCULAR MÉTRICAS
# ============================================
P  = TP / (TP + FP) if (TP + FP) > 0 else 0
R  = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

# ============================================
# 7. EXIBIR RESULTADOS VISUAIS
# ============================================
plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(mascara, cmap='gray')
plt.title(f'Máscara Gerada (T={limiar})')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gt_binaria, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(1, 4, 4)
sobreposicao = np.zeros((*mascara.shape, 3), dtype=np.uint8)
sobreposicao[(mascara == 1) & (gt_binaria == 1)] = [0, 255, 0]   # TP = Verde
sobreposicao[(mascara == 1) & (gt_binaria == 0)] = [255, 0, 0]   # FP = Vermelho
sobreposicao[(mascara == 0) & (gt_binaria == 1)] = [0, 0, 255]   # FN = Azul
plt.imshow(sobreposicao)
plt.title('TP (Verde) | FP (Vermelho) | FN (Azul)')
plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================
# 8. REPORTAR MÉTRICAS
# ============================================
print("=" * 50)
print("RESULTADOS DA SEGMENTAÇÃO")
print("=" * 50)
print(f"Limiar utilizado: {limiar}")
print("-" * 50)
print(f"TP (Verdadeiros Positivos):  {TP}")
print(f"FP (Falsos Positivos):       {FP}")
print(f"FN (Falsos Negativos):       {FN}")
print("-" * 50)
print(f"Precisão (P):    {P:.4f}  ({P*100:.2f}%)")
print(f"Recall (R):      {R:.4f}  ({R*100:.2f}%)")
print(f"F1-Score:        {F1:.4f}  ({F1*100:.2f}%)")
print("=" * 50)