import cv2
import matplotlib.pyplot as plt

caminho = 'imagem21.jpg'
img_colorida = cv2.imread(caminho)

if img_colorida is None:
    print("Erro")
else:
    
    img_verde = img_colorida[:, :, 1]

    hist = cv2.calcHist([img_verde], [0], None, [256], [0, 256])


    plt.figure(" Retinopatia ", figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_verde, cmap='gray')
    plt.title("Canal Verde (Green Channel)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(hist, color='green')
    plt.title("Histograma de Intensidades")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Quantidade de Pixels")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
    
    #3.1