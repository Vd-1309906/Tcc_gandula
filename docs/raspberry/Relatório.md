Relatório de Tecnologias e Métodos Aplicados
Este documento resume as tecnologias e as estratégias de otimização utilizadas no desenvolvimento do modelo de classificação de imagens, documentando a evolução do projeto desde a prova de conceito até as técnicas avançadas de refinamento.

Tabela 1: Tecnologias Base Aplicadas
Tecnologia / Ferramenta# Relatório de Tecnologias e Métodos Aplicados

Este documento resume as tecnologias e as estratégias de otimização utilizadas no desenvolvimento do modelo de classificação de imagens, documentando a evolução do projeto desde a prova de conceito até as técnicas avançadas de refinamento.

## Tabela 1: Tecnologias Base Aplicadas

| Tecnologia / Ferramenta | Versão / Tipo | Propósito no Projeto |
| :--- | :--- | :--- |
| **`Python`** | 3.11 | Linguagem de programação principal para o desenvolvimento de todos os scripts. |
| **`TensorFlow` / `Keras`** | 2.x | Framework principal para construção, treinamento e salvamento dos modelos de Deep Learning. |
| **`Scikit-learn`** | Mais recente | Utilizado para a avaliação de performance do modelo, gerando métricas detalhadas como Acurácia, Precisão, Recall e F1-Score. |
| **`NumPy`** | Mais recente | Biblioteca essencial para manipulação de dados numéricos (arrays), especialmente para processar as previsões do modelo. |
| **`TensorFlow Lite (TFLite)`**| - | Formato de modelo de destino, otimizado para implantação em dispositivos de borda com recursos limitados. |
| **Ambiente Virtual (`venv`)** | - | Ferramenta utilizada para isolar as dependências do projeto, garantindo a reprodutibilidade e evitando conflitos de pacotes. |

## Tabela 2: Métodos de Melhoria de Acurácia (em Ordem Cronológica)

| Método Aplicado | Problema Solucionado | Detalhes da Implementação e Impacto |
| :--- | :--- | :--- |
| **1. Transfer Learning com `MobileNetV2`** | Necessidade de um ponto de partida eficiente sem um dataset massivo. | Utilizamos um modelo `MobileNetV2` pré-treinado na `ImageNet`, congelando suas camadas e treinando apenas um novo classificador. **Resultado inicial:** Baixa acurácia (36%) devido ao modelo "viciado" em prever uma única classe. |
| **2. Data Augmentation** | Baixa quantidade de dados de treino e overfitting inicial. | Implementamos uma camada `Sequential` com `RandomFlip`, `RandomRotation` e `RandomZoom` para criar novas amostras de treino em tempo real. **Impacto:** Forçou o modelo a generalizar melhor e foi fundamental para sair do platô inicial. |
| **3. Aumento de Épocas e Coleta de Dados** | O modelo não tinha tempo/exemplos suficientes para aprender. | O número de épocas de treinamento foi aumentado (de 10 para 30+) e o dataset foi expandido (de ~90 para 500+ imagens). **Impacto:** Aumentou significativamente a base de conhecimento do modelo, permitindo que ele atingisse um novo patamar de acurácia (~70%). |
| **4. Callbacks (`EarlyStopping`, `ReduceLROnPlateau`)** | Treinamentos longos, risco de overfitting e otimização da taxa de aprendizado. | Implementamos `EarlyStopping` para parar o treino quando a performance parava de melhorar e `ReduceLROnPlateau` para ajustar a taxa de aprendizado dinamicamente. **Impacto:** Tornou o processo de treinamento mais eficiente e inteligente. |
| **5. Fine-Tuning (Ajuste Fino)** | Necessidade de especializar o modelo pré-treinado nos detalhes do nosso dataset. | Após o treino inicial, as últimas 30 camadas do modelo base foram "descongeladas" e o treinamento continuou com uma taxa de aprendizado muito baixa. **Impacto:** Permitiu que o modelo refinasse suas características, sendo uma das técnicas mais eficazes para ganhos de acurácia. |
| **6. Troca de Arquitetura (`ResNet50V2`)** | Tentativa de superar o platô de ~70% com um modelo mais poderoso. | O `MobileNetV2` foi substituído pelo `ResNet50V2`. Isso introduziu a necessidade de usar uma camada `Lambda` para o pré-processamento específico do modelo e `custom_objects` ao carregá-lo. **Impacto:** Demonstrou a flexibilidade do pipeline e a importância de testar diferentes arquiteturas. |
| **7. Análise de Erros** | Dificuldade em identificar por que o modelo falhava em certas previsões. | Criamos um script para salvar automaticamente as imagens classificadas incorretamente em pastas organizadas por tipo de erro (ex: `real_bola__previsto_nada`). **Impacto:** Moveu o foco para uma abordagem Data-Centric, permitindo a identificação visual de padrões de erro e a melhoria da qualidade do dataset. |

## Conclusão

O desenvolvimento do modelo seguiu uma trajetória iterativa e baseada em evidências. Cada platô de acurácia foi superado pela aplicação de uma nova técnica, começando com as mais simples (mais épocas) e progredindo para as mais complexas e eficazes (`Fine-Tuning` e `Análise de Erros`). Embora arquiteturas mais complexas como a `ResNet50V2` tenham sido exploradas, o modelo final escolhido foi o **`MobileNetV2`**. Esta decisão foi estratégica, priorizando a velocidade de inferência e a eficiência computacional, características essenciais para a implantação em dispositivos de borda com poder de processamento limitado, como um `Raspberry Pi`.

Versão / Tipo

Propósito no Projeto

Python

3.11

Linguagem de programação principal para o desenvolvimento de todos os scripts.

TensorFlow / Keras

2.x

Framework principal para construção, treinamento e salvamento dos modelos de Deep Learning.

Scikit-learn

Mais recente

Utilizado para a avaliação de performance do modelo, gerando métricas detalhadas como Acurácia, Precisão, Recall e F1-Score.

NumPy

Mais recente

Biblioteca essencial para manipulação de dados numéricos (arrays), especialmente para processar as previsões do modelo.

TensorFlow Lite (TFLite)

-

Formato de modelo de destino, otimizado para implantação em dispositivos de borda com recursos limitados.

Ambiente Virtual

venv

Ferramenta utilizada para isolar as dependências do projeto, garantindo a reprodutibilidade e evitando conflitos de pacotes.

Tabela 2: Métodos de Melhoria de Acurácia (em Ordem Cronológica)
Método Aplicado

Problema Solucionado

Detalhes da Implementação e Impacto

1. Transfer Learning com MobileNetV2

Necessidade de um ponto de partida eficiente sem um dataset massivo.

Utilizamos um modelo MobileNetV2 pré-treinado na ImageNet, congelando suas camadas e treinando apenas um novo classificador. Resultado inicial: Baixa acurácia (36%) devido ao modelo "viciado" em prever uma única classe.

2. Data Augmentation

Baixa quantidade de dados de treino e overfitting inicial.

Implementamos uma camada Sequential com RandomFlip, RandomRotation e RandomZoom para criar novas amostras de treino em tempo real. Impacto: Forçou o modelo a generalizar melhor e foi fundamental para sair do platô inicial.

3. Aumento de Épocas e Coleta de Dados

O modelo não tinha tempo/exemplos suficientes para aprender.

O número de épocas de treinamento foi aumentado (de 10 para 30+) e o dataset foi expandido (de ~90 para 500+ imagens). Impacto: Aumentou significativamente a base de conhecimento do modelo, permitindo que ele atingisse um novo patamar de acurácia (~70%).

4. Callbacks (EarlyStopping, ReduceLROnPlateau)

Treinamentos longos, risco de overfitting e otimização da taxa de aprendizado.

Implementamos EarlyStopping para parar o treino quando a performance parava de melhorar e ReduceLROnPlateau para ajustar a taxa de aprendizado dinamicamente. Impacto: Tornou o processo de treinamento mais eficiente e inteligente.

5. Fine-Tuning (Ajuste Fino)

Necessidade de especializar o modelo pré-treinado nos detalhes do nosso dataset.

Após o treino inicial, as últimas 30 camadas do modelo base foram "descongeladas" e o treinamento continuou com uma taxa de aprendizado muito baixa. Impacto: Permitiu que o modelo refinasse suas características, sendo uma das técnicas mais eficazes para ganhos de acurácia.

6. Troca de Arquitetura (ResNet50V2)

Tentativa de superar o platô de ~70% com um modelo mais poderoso.

O MobileNetV2 foi substituído pelo ResNet50V2. Isso introduziu a necessidade de usar uma camada Lambda para o pré-processamento específico do modelo e custom_objects ao carregá-lo. Impacto: Demonstrou a flexibilidade do pipeline e a importância de testar diferentes arquiteturas.

7. Análise de Erros

Dificuldade em identificar por que o modelo falhava em certas previsões.

Criamos um script para salvar automaticamente as imagens classificadas incorretamente em pastas organizadas por tipo de erro (ex: real_bola__previsto_nada). Impacto: Moveu o foco para uma abordagem Data-Centric, permitindo a identificação visual de padrões de erro e a melhoria da qualidade do dataset.

Conclusão
O desenvolvimento do modelo seguiu uma trajetória iterativa e baseada em evidências. Cada platô de acurácia foi superado pela aplicação de uma nova técnica, começando com as mais simples (mais épocas) e progredindo para as mais complexas e eficazes (Fine-Tuning e Análise de Erros). Embora arquiteturas mais complexas como a ResNet50V2 tenham sido exploradas, o modelo final escolhido foi o MobileNetV2. Esta decisão foi estratégica, priorizando a velocidade de inferência e a eficiência computacional, características essenciais para a implantação em dispositivos de borda com poder de processamento limitado, como um Raspberry Pi.
