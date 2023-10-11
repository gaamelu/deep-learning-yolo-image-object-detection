# Projeto computacional
> Trabalho desenvolvido por **Gabriel Melo Lucena (RA 184254)** para a disciplina **MS960** de **Aprendizado de Máquina Profundo** do 2º semestre de 2023.

O trabalho foi desenvolvido em **python** incluindo as *rotinas computacionais* e um relatório contendo *discussão crítica sobre os métodos e resultados* do projeto. A arquitetura inicial foi preparada pelo professor da disciplina, **Prof. João Batista Florindo**.

## O problema de detecção de objetos

‘O *problema de detecção de objetos* consiste em identificar se há uma ou mais classes de objetos e suas localizações, dado uma imagem de input. Uma solução ótima para esse problema pode ser implementado em diversas situações, inclusive aplicado à Unicamp. As câmeras do **Restaurante Universitário** utilizam detecção de objetos, em especial de pessoas, para [observar o fluxo de pessoas](https://www.unicamp.br/unicamp/noticias/2021/06/09/cameras-dimensionam-o-fluxo-nos-restaurantes-universitarios) durante o funcionamento das refeições. Outras aplicações incluem detecção de objetos para tomada de decisão de carros autônomos. Uma adaptação mais complexa do problema de detecção de objetos, voltados para pessoas, inclui um exemplo de identificação de pessoas por ondas de rádio que identifica pessoas e movimentações através de paredes, reconhecendo a figura e a movimentação da pessoa, um [projeto desenvolvido pelo MIT](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2406.pdf).

Uma imagem pode ser considerada como um dado não estruturado, o que na área de aprendizado de máquina é melhor lidado com *redes neurais* ao invés de outros algoritmos tradicionais de aprendizado de máquina. Mais especificamente, é comum o output da solução ser uma bounding box (região retangular na imagem) contendo o objeto e a confiança do algoritmo que esse objeto é da classe indicada.

Uma técnica simples para o problema chamada de **sliding window** (janela deslizante), é escolher uma bounding box, então percorrer toda a imagem, criando uma classificação para cada análise, de forma que vamos passar secções da imagem várias e várias vezes para as classificações. Um problema desse método, é o tamanho da bounding box. Um objeto na imagem pode estar em diversas escalas distintas, o que torna problemático esse método.

Uma **rede neural convolucional** é adequada para problemas que usem dados que as vizinhanças das features sejam importantes e também para problemas de transitividade, em que as features possam transladar em relação ao todo.

As soluções ao problema normalmente se dividem em:

### Extração de características por CNN
Um exemplo é a **R-CNN** que faz uma busca seletiva para extrair uma quantidade de candidatos para calcular as características de cada um por CNN. Esse é um método considerado bem lento, o que gerou a necessidade de desenvolver variações mais rápidas aplicando novas técnicas na rede neural, que são **Fast-R-CNN** e **Faster-R-CNN**.

É interessante ressaltar que as redes neurais podem ser construídas com objetivos diferentes, como identificar os objetos em si ou contar quantos objetos de cada classe se encontram na imagem.

### Identificação por “observação única” com CNN

O exemplo em questão é a **YOLO** (**Y**ou **O**nly **L**ook **O**nce) em que consiste em pegar a imagem de input com um grid, então gerar várias bounding boxes (por algum critério) e então escolher as bounding box com maiores probabilidades de serem objeto a classificar. Um problema comum da rede YOLO é que ela tem dificuldade de identificar objetos realmente pequenos. Outro método de “observação única” é o **Single Shot Detector (SSD)**.

## O algoritmo YOLOv3

O algoritmo YOLO é muito usado para detecção de objetos em tempo real, mantendo uma boa precisão e eficiência. Em algumas de suas técnicas aplicadas, permite uma boa capacidade de aprender os objetos em diferentes escalas e aspectos. Sua arquitetura recebe como entrada uma imagem com escala padronizada, então como output é classificada entre as classes que o modelo foi treinado. Um dos passos da rede neural consiste em criar escalas diferentes das estimativas de regiões de interesse (anchor boxes) em que serão retornadas no output, a rede retorna uma matriz com parâmetros $[x-topleft, y-topleft, width, height, p1, p2, p3, …, p80]$ para cada bounding box de cada anchor box.

## Implementação do algoritmo YOLO

Iniciaremos as bibliotecas utilizadas

```python
# Leitura de bibliotecas
import cv2 as cv2 # opencv
import numpy as np
import random
```

Iniciamos os parâmetros do algoritmo, como dimensões padronizada do algoritmo, o diretório da imagem de teste e os valores **confidence_threashold** e **non_maximal_suppresion_threshold**.

```python
# Parâmetros da rede neural
confidence_threshold = 0.5
non_maximal_supression_threshold = 0.1

# Dimensões da imagem
image_width = 1024
image_height = 1024

# Diretório da imagem
image_path = "images\dog2.jpg"
```

Carregamos a imagem de teste, redimensionamos, lemos os títulos de cada classe e processamos a imagem no padrão do input da rede neural.

```python
# Lê a imagem
frame = cv2.imread(image_path)

# Redimensiona a imagem
frame = cv2.resize(frame, (image_width, image_height))

# Leitura das classes
with open("data/coco.names", 'rt') as f:
    classes = f.read().splitlines()

# Cria o blob 4D da imagem
blob = cv2.dnn.blobFromImage(frame, 1/255, (image_width, image_height), [0,0,0], 1, crop=False)
```

Construímos a rede neural a partir da estrutura da rede neural em **.cfg** e dos pesos pre-processados em **.weights**, então executamos o forward da rede com o input oferecido. Essa estrutura foi discutida anteriormente e aplicada a seguir:

```python
# Carrega a rede neural YOLOv3
net = cv2.dnn.readNetFromDarknet("cfg/yolov3.cfg", "weights/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Define o input da rede neural
net.setInput(blob)

# Carrega o nome das camadas da rede
layersNames = net.getLayerNames()

# Obtem o nome das camadas de output da rede
outputLayers = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Executa o passo forward da rede para obter o output
outs = net.forward(outputLayers)
```

Nesse momento, temos o resultado da rede neural como todas as bounding box identificadas. Porém, são muitas bounding box identificadas, inclusive várias para o mesmo objeto ou áreas semelhantes.

```python
initial_boxes = []

for output in outputs:
    for detection in output:
        # detection = [x, y, width, height, p1, p2, p3, …, p80]
        # sendo p1 = prob da classe 1, p2 = prob da classe 2, …, p80 = prob da classe 80
        scores = detection[5:] # lista de probabilidades das classes 1 a 80
        classId = np.argmax(scores)
        confidence = float(scores[classId])

        center_x = int(detection[0] * image_height)
        center_y = int(detection[1] * image_height)
        width = int(detection[2] * image_height)
        height = int(detection[3] * image_height)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        
        initial_boxes.append({"confidence": confidence, "classId": classId, "left": left, "top": top, "width": width, "height": height})
```

> [!info] Definição de Intersection over Union $IoU$
> Um método que calcula a razão da intersecção sobre a união das bounding box. Em outras palavras, obtemos um parâmetro da “intensidade” com que as bounding box se sobrepõem, logo, identificam o mesmo objeto.

Então vamos aplicar o método **Non-Max Suppression**. Primeiramente, nós filtramos todas as bounding box com confiança de inferior ao parâmetro **confidence_threshold**, então identificamos a $IoU$ entre as bounding box e escolhemos a bounding box de maior parâmetro de confiança $p_{c}$. Em seguida, filtramos todos as bounding box com alto valor de $IoU$ e valor inferior a $p_{c}$ em relação à bounding box escolhida. Seguimos esse próximo várias vezes até que não haja mais bounding box para remover.

```python
def calculate_iou(box1, box2):
    # Extrai as coordenadas das bounding boxes
    x1, y1, w1, h1 = box1["left"], box1["top"], box1["width"], box1["height"]
    x2, y2, w2, h2 = box2["left"], box2["top"], box2["width"], box2["height"]
    
    # Calcula as coordenadas dos pontos de interseção
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Calcula a área da interseção
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calcula a área da união
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calcula o IoU
    iou = intersection_area / union_area
    
    return iou

initial_boxes.sort(key=lambda x: x["confidence"], reverse=True)

initial_boxes = [box for box in initial_boxes if box["confidence"] >= confidence_threshold]

boxes = []
while len(initial_boxes):
    current_box = initial_boxes.pop(0)
    boxes.append(current_box)
    initial_boxes = [box for box in initial_boxes if calculate_iou(current_box, box) < non_maximal_supression_threshold]
```

Nesse momento, nós temos os melhores candidatos a objetos reconhecidos. Então, vamos desenhar as bounding boxes na imagem original redimensionada:

```python
classColors = {}
for bounding_box in boxes:
    # Extraimos as informações
    left, top, width, height = bounding_box["left"], bounding_box["top"], bounding_box["width"], bounding_box["height"]

    object_class = classes[bounding_box["classId"]]
    object_confidence = bounding_box["confidence"]

    # Escolhemos uma cor para a classe
    color = classColors[object_class] if object_class in classColors else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    classColors[object_class] = color

    # Desenhamos a própria boundind box
    cv2.rectangle(frame, (left, top), (left+width, top+height), color, 3)

    # Desenhamos a classe e a confiança

    label = f"{object_class} {int(object_confidence * 100)}%"
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 1
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)

    x, y = left, top - text_height - 5

    cv2.rectangle(frame, (x - 3, y - 5), (x + text_width + 5, y + text_height + 5), color, -1)
    cv2.putText(frame, label, (x, y + text_height), font, font_scale, (0, 0, 0), 2)

cv2.imwrite("YOLOv3 output.jpg", frame.astype(np.uint8))
```

No fim, teremos uma imagem escrita em **YOLOv3 output.jpg** com as bounding boxes com as classificações dos objetos identificados.

## Resultados

Como referência, mostrarei todas as imagens para um dos melhores parâmetros que encontrei, então mostrarei imagens pontuais com diferenças significativas em relação à referência.

### Implementação para geração de todas as imagens de teste

```python
# Implementação geral para múltiplas imagens

import os
import shutil

# Define o caminho da pasta de entrada
input_folder = "images/"

# Parâmetros da rede neural
confidence_threshold = 0.5
non_maximal_supression_threshold = 0.1

output_folder = f"output/confidence{confidence_threshold}_nms{non_maximal_supression_threshold}"

# Dimensões da imagem

image_width = 512
image_height = 512

# Cria a pasta de saída, se ela não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carrega a rede neural YOLOv3
net = cv2.dnn.readNetFromDarknet("cfg/yolov3.cfg", "weights/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Carrega o nome das camadas da rede
layersNames = net.getLayerNames()

# Obtem o nome das camadas de output da rede
outputLayers = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

classColors = {}

# Percorre todos os arquivos da pasta de entrada
for image_path in os.listdir(input_folder):
    # Verifica se o arquivo é uma imagem
    # Define o caminho completo do arquivo de entrada e do arquivo de saída

    # Lê a imagem
    frame = cv2.imread(os.path.join(input_folder, image_path))

    # Redimensiona a imagem
    frame = cv2.resize(frame, (image_width, image_height))

    # Leitura das classes
    with open("data/coco.names", 'rt') as f:
        classes = f.read().splitlines()

    # Cria o blob 4D da imagem
    blob = cv2.dnn.blobFromImage(frame, 1/255, (image_width, image_height), [0,0,0], 1, crop=False)
    
    # Define o input da rede neural
    net.setInput(blob)
    # Executa o passo foward da rede para obter o output
    outputs = net.forward(outputLayers)

    initial_boxes = []

    for output in outputs:
        for detection in output:
            # detection = [x, y, width, height, p1, p2, p3, ..., p80]
            # sendo p1 = prob da classe 1, p2 = prob da classe 2, ..., p80 = prob da classe 80
            scores = detection[5:] # lista de probabilidades das classes 1 a 80
            classId = np.argmax(scores)
            confidence = float(scores[classId])

            center_x = int(detection[0] * image_height)
            center_y = int(detection[1] * image_height)
            width = int(detection[2] * image_height)
            height = int(detection[3] * image_height)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            
            initial_boxes.append({"confidence": confidence, "classId": classId, "left": left, "top": top, "width": width, "height": height})

    initial_boxes.sort(key=lambda x: x["confidence"], reverse=True)

    initial_boxes = [box for box in initial_boxes if box["confidence"] >= confidence_threshold]

    boxes = []
    while len(initial_boxes):
        current_box = initial_boxes.pop(0)
        boxes.append(current_box)
        initial_boxes = [box for box in initial_boxes if calculate_iou(current_box, box) < non_maximal_supression_threshold]

    for bounding_box in boxes:
        # Extraimos as informações
        left, top, width, height = bounding_box["left"], bounding_box["top"], bounding_box["width"], bounding_box["height"]

        object_class = classes[bounding_box["classId"]]
        object_confidence = bounding_box["confidence"]

        # Escolhemos uma cor para a classe
        color = classColors[object_class] if object_class in classColors else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        classColors[object_class] = color

        # Desenhamos a própria boundind box
        cv2.rectangle(frame, (left, top), (left+width, top+height), color, 3)

        # Desenhamos a classe e a confiança

        label = f"{object_class} {int(object_confidence * 100)}%"
        font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 1
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)

        x, y = left, top - text_height - 5

        cv2.rectangle(frame, (x - 3, y - 5), (x + text_width + 5, y + text_height + 5), color, -1)
        cv2.putText(frame, label, (x, y + text_height), font, font_scale, (0, 0, 0), 2)

    cv2.imwrite(os.path.join(output_folder, image_path), frame.astype(np.uint8))
```

#### confidence_threshold = 0.5, non_maximal_supression_threshold = 0.1
Para esses parâmetros, notamos uma boa capacidade de identificação de objetos, especialmente objetos maiores e isolados, porém também obteve uma boa capacidade de identificar objetos menores ou muito próximos.

##### Imagens
![[README_image_1.jpg]]

![[README_image_2.jpg]]

![[README_image_3.jpg]]

![[README_image_4.jpg]]

![[README_image_5.jpg]]

![[README_image_6.jpg]]

![[README_image_7.jpg]]

![[README_image_8.jpg]]

![[README_image_9.jpg]]

![[README_image_10.jpg]]

![[README_image_11.jpg]]

![[README_image_12.jpg]]

#### confidence_threshold = 0.2, non_maximal_supression_threshold = 0.4
Com essa redução, o algoritmo não identificou o cachorro
![[README_image_13.png]]
![[README_image_14.png]]
E nem a zebra.
![[README_image_15.png]]
E nem a moto.

#### confidence_threshold = 0.2, non_maximal_supression_threshold = 0.5
A maior diferença significativa nesses parâmetros, foi uma má identificação da coluna do ambiente com uma televisão
![[README_image_16.png]]

#### confidence_threshold = 0.8, non_maximal_supression_threshold = 0.5

![[README_image_17.png]]

E também perdeu capacidade de identificar as pessoas ao fundo.
![[README_image_18.png]]

### Conclusão dos resultados
Em geral, podemos concluir que **o *algoritmo YOLO* funcionou bem no problema de identificação dos objetos**. Em especial, observamos que quanto menor o parâmetro **confidence_threshold**, mais margem de errar a classificação das imagens, enquanto que quanto maior o parâmetro **non_maximal_supresion_threshold**, menor capacidade de distinguir objetos muito próximos.  

# Referências
- Notas de aula da semana 12 da disciplina MS960, prof. Florindo.
- [Como funciona a Detecção de Objetos na Prática?](https://didatica.tech/deteccao-de-objetos/)
- [Deep Learning::Detecção de Objetos em Imagens](https://lapix.ufsc.br/ensino/visao/visao-computacionaldeep-learning/deteccao-de-objetos-em-imagens/)
- [Zero to Hero: Guide to Object Detection using Deep Learning: Faster R-CNN,YOLO,SSD – CV-Tricks.com](https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/)
- [Site Unreachable](https://ichi.pro/pt/deteccao-de-objetos-e-segmentacao-de-instancias-uma-visao-geral-detalhada-163595582338290)
- [ww2.inf.ufg.br/\~anderson/deeplearning/20181/Curso\_DeepLearning - Object Detection- SSD Fast Faster RCNN Yolo.pdf](https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Curso_DeepLearning%20-%20Object%20Detection-%20SSD%20Fast%20Faster%20RCNN%20Yolo.pdf)
- [Detecção de Objetos com YOLO - Uma abordagem moderna | IA Expert Academy](https://iaexpert.academy/2020/10/13/deteccao-de-objetos-com-yolo-uma-abordagem-moderna/)
- [YOLOv3 – Deep Learning Based Object Detection – YOLOv3 with OpenCV ( Python / C++ )](https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/)
- [Non-Maximum Suppression with cv2.dnn.NMSBoxes: A Comprehensive Guide - Program Talk](https://programtalk.com/python/non-maximum-suppression-with-cv2-dnn-nmsboxes-a-comprehensive-guide/)
- [Error](https://ieeexplore.ieee.org/document/8839032)
