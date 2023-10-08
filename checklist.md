# Projeto Computacional I
## Reconhecimento de objetos usando YOLO

A atividade consiste em aplicar o algoritmo YOLO em detecção de imagens.

### Observações
- Na pasta **images/** está uma série de imagens de exemplo para testar o algoritmo.
- O arquivo **cfg/yolov3.cfg** contém a *v3 da arquitetura de rede usada pelo YOLO*. (**Observação**: não adicionado ao GitHub por ser muito grande!)
- [ ] Buscar implementar a versão mais atual da arquitetura
- O arquivo **weights/yolov3.weights** contém os pesos pré-treinados.
- Em **data/coco.names** contém a lista de 80 classes de objetos usadas no treinamento de detecção.

### Instruções do projeto
- [ ] Mostrar a arquitetura da rede neural do algoritmo YOLO.
- [ ] Implementar o carregamento, redimensionamento e exibição correta das imagens.
- [ ] Implementar e testar diferentes valores para o *Limiar de Supressão Não-Maximal* e para a *Intersecção Sobre a União*.
- Implementar a detecção de objetos mostrando o número de objetos detectados e os próprios objetos com seus níveis de confiança em suas bounding boxes.

### Instruções do relatório
- [ ] Introduzir o problema de detecção de objetos, sua importância e principais métodos usados na atualidade.
- [ ] Descrever os passos e a arquitetura do algoritmo YOLO e a intuição/motivação dos passos. Resuma os aspectos teóricos e práticos do algoritmo.
- [ ] Mostrar resultados nas imagens de base anexa e imagens de escolha.
- [ ] Discutir os resultados, formular hipóteses que expliquem os resultados, comentar sobre o papel dos limiares, discutir possíveis melhorias, etc.
- [ ] Listar referências usadas na escrita do relatório e na produção do seu código.
