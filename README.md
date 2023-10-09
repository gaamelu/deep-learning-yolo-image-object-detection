# Projeto computacional
> Trabalho desenvolvido por **Gabriel Melo Lucena (RA 184254)** para a disciplina **MS960** de **Aprendizado de Máquina Profundo** do 2º semestre de 2023.

O trabalho foi desenvolvido em **python** incluindo as *rotinas computacionais* e um relatório contendo *discussão crítica sobre os métodos e resultados* do projeto. A arquitetura inicial foi preparada pelo professor da disciplina, **Prof. João Batista Florindo**.

## O problema de detecção de objetos

‘O *problema de detecção de objetos* consiste em identificar se há uma ou mais classes de objetos e suas localizações, dado uma imagem de input. Uma imagem pode ser considerada como um dado não estruturado, o que na área de aprendizado de máquina é melhor lidado com *redes neurais* ao invés de outros algoritmos tradicionais de aprendizado de máquina. Mais especificamente, é comum o output da solução ser uma bounding box (região retangular na imagem) contendo o objeto e a confiança do algoritmo que esse objeto é da classe indicada.

Uma técnica simples para o problema chamada de **sliding window** (janela deslizante), é escolher uma bounding box, então percorrer toda a imagem, criando uma classificação para cada análise, de forma que vamos passar secções da imagem várias e várias vezes para as classificações. Um problema desse método, é o tamanho da bounding box. Um objeto na imagem pode estar em diversas escalas distintas, o que torna problemático esse método.

Uma **rede neural convolucional** é adequada para problemas que usem dados que as vizinhanças das features sejam importantes e também para problemas de transitividade, em que as features possam transladar em relação ao todo.

As soluções ao problema normalmente se dividem em:

### Extração de características por CNN
Um exemplo é a **R-CNN** que faz uma busca seletiva para extrair uma quantidade de candidatos para calcular as características de cada um por CNN. Esse é um método considerado bem lento, o que gerou a necessidade de desenvolver variações mais rápidas aplicando novas técnicas na rede neural, que são **Fast-R-CNN** e **Faster-R-CNN**.

É interessante ressaltar que as redes neurais podem ser construídas com objetivos diferentes, como identificar os objetos em si ou contar quantos objetos de cada classe se encontram na imagem.

### Identificação por “observação única” com CNN

O exemplo em questão é a **YOLO** (**Y**ou **O**nly **L**ook **O**nce) em que consiste em pegar a imagem de input com um grid, então gerar várias bounding boxes (por algum critério) e então escolher as bounding box com maiores probabilidades de serem objeto a classificar. Um problema comum da rede YOLO é que ela tem dificuldade de identificar objetos realmente pequenos. Outro método de “observação única” é o **Single Shot Detector (SSD)**.

## O algoritmo YOLO
- [ ] Descrever os passos e a arquitetura do algoritmo YOLO e a intuição/motivação dos passos. Resuma os aspectos teóricos e práticos do algoritmo.

# Referências
- Notas de aula da semana 12 da disciplina MS960, prof. Florindo.
- [Como funciona a Detecção de Objetos na Prática?](https://didatica.tech/deteccao-de-objetos/)
- [Deep Learning::Detecção de Objetos em Imagens](https://lapix.ufsc.br/ensino/visao/visao-computacionaldeep-learning/deteccao-de-objetos-em-imagens/)
- [Zero to Hero: Guide to Object Detection using Deep Learning: Faster R-CNN,YOLO,SSD – CV-Tricks.com](https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/)
- [Site Unreachable](https://ichi.pro/pt/deteccao-de-objetos-e-segmentacao-de-instancias-uma-visao-geral-detalhada-163595582338290)
- [ww2.inf.ufg.br/\~anderson/deeplearning/20181/Curso\_DeepLearning - Object Detection- SSD Fast Faster RCNN Yolo.pdf](https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Curso_DeepLearning%20-%20Object%20Detection-%20SSD%20Fast%20Faster%20RCNN%20Yolo.pdf)
- [Detecção de Objetos com YOLO - Uma abordagem moderna | IA Expert Academy](https://iaexpert.academy/2020/10/13/deteccao-de-objetos-com-yolo-uma-abordagem-moderna/)
