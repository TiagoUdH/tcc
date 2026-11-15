# Estrutura dos capítulos de Resultados e Discussão

Cada seção indica as tabelas e figuras correspondentes (arquivos em `resultados/`) e os números
principais que o texto deve citar. Todos os valores de F1 são F1-Macro médio de 3 seeds
(42, 123, 2024) no conjunto de teste.

---

## Capítulo 4 — Resultados

### 4.1 Desempenho comparativo dos modelos

Apresentar a comparação geral antes de detalhar qualquer modelo.

- **Tabela 4.1** — Desempenho por modelo e dataset (acurácia, F1, precisão, recall, média ± dp).
  Fonte: [consolidado/resultados_comparativos.csv](resultados/consolidado/resultados_comparativos.csv)
- **Figura 4.1** — Mapa de calor F1 (modelo × dataset).
  Fonte: [consolidado/matriz_avaliacao_cruzada_f1.png](resultados/consolidado/matriz_avaliacao_cruzada_f1.png)

| Modelo | BRACOL | Coffee Leaf | RoCoLe | JMuBEN |
|---|---|---|---|---|
| EfficientNetV2B0 | **0,967** | **0,996** | **0,739** | — |
| ViT | 0,962 | 0,983 | 0,737 | 1,000 |
| MobileNetV3Large | 0,942 | 0,934 | 0,618 | — |
| MobileNetV3Small | 0,937 | 0,929 | 0,609 | — |
| MobileNetV2 | 0,924 | 0,880 | 0,656 | — |
| NASNetMobile | 0,899 | 0,847 | 0,495 | — |
| Random Forest | 0,520 | 0,516 | 0,432 | 1,000 |

Pontos a descrever:
- EfficientNetV2B0 tem o maior F1 nos três datasets em que foi avaliado; ViT fica logo atrás.
- RoCoLe é o dataset mais difícil (melhor F1 = 0,739); Coffee Leaf, o mais fácil entre os três.
- NASNetMobile é a CNN mais fraca e a de maior variância entre seeds.

### 4.2 Testes de significância estatística

- **Tabela 4.2** — Pares relevantes com diferença média e p-valor (teste t pareado entre seeds).
  Fonte: [consolidado/testes_significancia.csv](resultados/consolidado/testes_significancia.csv)
  (são 64 pares; na tabela do texto, mostrar só os pares do EfficientNetV2B0 e o resto no apêndice).

Pontos a descrever:
- EfficientNetV2B0 × ViT **não** difere significativamente em nenhum dataset
  (p = 0,52 BRACOL; 0,07 Coffee Leaf; 0,96 RoCoLe).
- EfficientNetV2B0 supera significativamente as demais CNNs na maioria dos pares
  (exceção: MobileNetV3Large no BRACOL, p = 0,074; MobileNetV2 no RoCoLe, p = 0,079).
- Random Forest perde significativamente para todos os modelos profundos, exceto
  NASNetMobile no RoCoLe (p = 0,055).
- O Wilcoxon aparece como N/A porque exige n ≥ 6; registrar isso aqui ou na metodologia.

### 4.3 Eficiência computacional e viabilidade embarcada

- **Tabela 4.3** — Tamanho do modelo exportado, latência em memória e latência exportada.
  Fonte: colunas de latência em [resultados_comparativos.csv](resultados/consolidado/resultados_comparativos.csv)
  e [consolidado/latencia_tflite.csv](resultados/consolidado/latencia_tflite.csv) (com p95).

Pontos a descrever (BRACOL):
- EfficientNetV2B0 TFLite: 6,9 MB, ~37 ms por imagem.
- ViT PyTorch Mobile INT8: 42,3 MB, ~53 ms → 6× maior e mais lento, com F1 equivalente.
- MobileNetV3Small: 1,2 MB e 4,7 ms — a opção mais leve, com F1 0,937.
- Random Forest: 0,9 MB e ~16 ms, mas com F1 0,52.
- Observação: latências medidas em sessões diferentes do Kaggle variam (~17%); comparar
  ordens de grandeza, não diferenças pequenas.

### 4.4 Análise detalhada do modelo escolhido (EfficientNetV2B0 × BRACOL)

#### 4.4.1 Desempenho por classe
- **Figura 4.2** — Matriz de confusão.
  [efficientnet_bracol/matriz_confusao_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/matriz_confusao_EfficientNetV2B0_bracol.png)
- **Tabela 4.4** — Precisão, recall e F1 por classe.
  [efficientnet_bracol/relatorio_classificacao_EfficientNetV2B0_bracol.csv](resultados/efficientnet_bracol/relatorio_classificacao_EfficientNetV2B0_bracol.csv)
- Figura opcional: [f1_por_classe_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/f1_por_classe_EfficientNetV2B0_bracol.png)

Números: saudável F1 1,000; ferrugem 0,990; phoma 0,960; bicho-mineiro 0,945;
cercosporiose 0,922 (classe mais difícil). Acurácia 0,966 no teste (409 imagens).

#### 4.4.2 Curvas ROC e Precisão-Recall
- **Figura 4.3** — [curvas_roc_pr_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/curvas_roc_pr_EfficientNetV2B0_bracol.png)

#### 4.4.3 Calibração das probabilidades
- **Figura 4.4** — Diagrama de confiabilidade antes/depois do *temperature scaling*.
  [calibracao_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/calibracao_EfficientNetV2B0_bracol.png),
  [calibracao_temperature_scaling_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/calibracao_temperature_scaling_EfficientNetV2B0_bracol.png)
- Citar a temperatura ajustada na validação: T = 1,539 (o modelo era excessivamente confiante).

#### 4.4.4 Predição seletiva (abstenção)
- **Figura 4.5** — Curva risco × cobertura.
  [risco_cobertura_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/risco_cobertura_EfficientNetV2B0_bracol.png)
- **Tabela 4.5** — [abstencao_EfficientNetV2B0_bracol.csv](resultados/efficientnet_bracol/abstencao_EfficientNetV2B0_bracol.csv)

Números: com limiar 0,9 o modelo responde 92,2% das imagens com 98,4% de acerto;
com 0,99, cobre 73,1% com 99,0% de acerto (dos 14 erros, só 3 passam).

### 4.5 Explicabilidade

#### 4.5.1 Análise qualitativa
- **Figura 4.6** — Painel comparando Grad-CAM, LIME e Integrated Gradients.
  [painel_xai_EfficientNetV2B0_bracol.png](resultados/efficientnet_bracol/painel_xai_EfficientNetV2B0_bracol.png)
- Figuras por classe para o apêndice: `gradcam_*`, `lime_*`, `ig_*` em
  [resultados/efficientnet_bracol/](resultados/efficientnet_bracol/).

#### 4.5.2 Avaliação quantitativa da fidelidade (inserção/remoção)
- **Figura 4.7** — Curvas médias de inserção e remoção, Grad-CAM × mapa aleatório.
  [fidelidade/gradcam_fidelidade_curvas.png](resultados/fidelidade/gradcam_fidelidade_curvas.png)
- **Tabela 4.6** — AUC de inserção, AUC de remoção, escore e p-valor (geral e por classe).
  [fidelidade/gradcam_fidelidade_resumo.csv](resultados/fidelidade/gradcam_fidelidade_resumo.csv),
  [fidelidade/gradcam_fidelidade_EfficientNetV2B0_bracol.csv](resultados/fidelidade/gradcam_fidelidade_EfficientNetV2B0_bracol.csv)

Números (100 imagens, 20 por classe):
- Inserção 0,954 (aleatório 0,873); remoção 0,431 (aleatório 0,611).
- Escore (inserção − remoção) 0,523 contra 0,261 do aleatório; Wilcoxon p = 1,4 × 10⁻¹⁴.
- Por classe: ferrugem 0,842, bicho-mineiro 0,612, cercosporiose 0,612, phoma 0,396 —
  todas significativas; **saudável não significativa** (p = 0,99).

### 4.6 Random Forest com características convolucionais (baseline da IC)

- **Figura 4.8** — Importância das características (MDI e permutação).
  [random_forest/RandomForest_bracol/importancia_features_RandomForest_bracol.png](resultados/random_forest/RandomForest_bracol/importancia_features_RandomForest_bracol.png)
- **Figura 4.9** — Matriz de confusão do RF.
  [random_forest/RandomForest_bracol/matriz_confusao_RandomForest_bracol.png](resultados/random_forest/RandomForest_bracol/matriz_confusao_RandomForest_bracol.png)
- Figuras opcionais: `shap_resumo_*`, `filtros_*`, `lime_*`, `risco_cobertura_*` na mesma pasta.

Números:
- Característica mais importante: desvio padrão do Sobel (MDI 0,30), seguida do desvio do
  Laplaciano e da média do Sobel → o RF depende de textura e bordas.
- Ferrugem é confundida com bicho-mineiro em 45 das 149 imagens.
- A confiança máxima raramente passa de ~0,85: com limiar 0,8 cobre só 10,5% das imagens.
- "Média (Laplaciano)" tem importância zero (ver 5.4).

### 4.7 Aplicativo móvel

- **Figuras 4.10–4.12** — Capturas de tela do app: escolha da imagem, classificação com
  probabilidades calibradas, mapas Grad-CAM/oclusão/LIME e curva de fidelidade.
  *(Pendente: tirar os prints no celular com o APK atual.)*
- Descrever o fluxo: TFLite com Grad-CAM embutido no grafo, *temperature scaling*,
  explicações calculadas no próprio aparelho (isolates), sem envio de dados à nuvem.
- Citar: modelo de 7,6 MB, inferência ~38 ms (medida no Kaggle; se possível, medir no celular).

---

## Capítulo 5 — Discussão

### 5.1 Redes profundas × características artesanais
- O salto de ~0,52 para ~0,97 de F1 mostra que estatísticas globais de filtros não capturam
  a localização e a forma das lesões.
- Ligar à importância das características (4.6): o RF só "vê" textura global, por isso confunde
  doenças de textura parecida (ferrugem × bicho-mineiro).

### 5.2 Escolha do EfficientNetV2B0 para o aplicativo
- Desempenho estatisticamente equivalente ao ViT (4.2) com 6× menos tamanho e menor latência (4.3).
- MobileNetV3Small como alternativa para aparelhos muito modestos (F1 0,937, 1,2 MB).
- Calibração + abstenção tornam o modelo mais seguro para o produtor (4.4.3–4.4.4).

### 5.3 O que as explicações mostram
- O Grad-CAM é fiel nas doenças: aponta regiões que de fato sustentam a decisão (4.5.2).
- Na classe saudável não há lesão a localizar, então a evidência fica espalhada pela folha
  e o mapa não supera o aleatório — resultado esperado, não falha do método.
- Phoma tem o menor escore: discutir lesões menores/mais difusas.

### 5.4 Limitações
- **Poucas seeds:** n = 3 dá baixo poder estatístico; o Wilcoxon não pôde ser aplicado (n < 6).
- **JMuBEN:** até o RF chega a F1 = 1,0, indicando que o dataset é fácil demais
  (provável diferença de condições de captura entre classes ou imagens quase duplicadas);
  por isso não foi usado para diferenciar as CNNs.
- **RoCoLe:** F1 máximo de 0,74 — classes desbalanceadas e mais parecidas entre si.
- **Feature constante no RF:** a média do Laplaciano (`ndimage.laplace`, borda refletida) é
  praticamente zero em toda imagem; o RF usa na prática 9 características.
- **Latência:** medida em GPU/CPU do Kaggle, não no celular; varia entre sessões.
- **Fidelidade:** avaliada só para o Grad-CAM do modelo escolhido, em 100 imagens do BRACOL.
- **Generalização:** os modelos foram treinados e testados dentro de cada dataset; não há teste
  com fotos de campo de outra origem.

### 5.5 Trabalhos futuros
- Mais seeds (≥ 6) para viabilizar o Wilcoxon.
- Testar o app em campo e medir latência em aparelhos reais.
- Treinar com datasets combinados e avaliar entre datasets.
- Estender a avaliação de fidelidade para LIME e IG e para os demais modelos.

---

## Apêndices sugeridos
- A: Tabela completa dos 64 testes de significância.
- B: Resultados por seed (`multiseed_*.csv`).
- C: Explicações por classe (Grad-CAM, LIME, IG) do EfficientNetV2B0.
- D: Histórico do Optuna e hiperparâmetros finais de cada modelo.
