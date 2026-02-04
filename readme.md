## 🏗️ Classes

### 1. `Watcher` 

* **Responsabilidade:** Inicializa a câmera (ou Thread), gerencia o loop principal, chama a inferência e comanda o gravador de dados. É ele quem conecta todas as outras classes.

### 2. `Inference` 

* **Função:** Decide **QUANDO** a IA deve rodar.
* **Responsabilidade:** Contém a máquina de estados (Modos 1, 2, 3, 4). Ele recebe o tempo e o frame atual e retorna `True` (pode processar) ou `False` (deve dormir).

### 3. `CameraThread` 

* **Função:** Leitura de imagem assíncrona.
* **Responsabilidade:** Roda em paralelo (background) capturando imagens infinitamente e descartando as velhas. Garante que, quando o Watcher pedir uma imagem, ele receba o frame mais recente possível (0ms de lag de input).

### 4. `FrameGate` (O Reflexo / Visão Periférica)

* **Responsabilidade:** Reduz a imagem para 32px, converte para cinza e compara com o quadro anterior. Se a mudança de pixels for maior que o `threshold` (0.015), ele avisa que houve movimento. Custa quase zero de CPU.

### 5. `YoloSensor`

* **Função:** Wrapper da Inteligência Artificial.
* **Responsabilidade:** Carrega o modelo YOLO, faz a inferência pesada e desenha as caixas delimitadoras (`bounding boxes`) na imagem.

### 6. `Recorder`

* **Responsabilidade:** Salva cada frame processado no arquivo `benchmark.csv`, calculando métricas como FPS de Hardware e detectando se há pessoas na cena.

### 7. `BenchmarkPlotter` 

* **Função:** Visualização de dados.
* **Responsabilidade:** Lê o CSV e gera gráficos de Estabilidade (Boxplot), Aquecimento (Lineplot) e Potencial (Barplot).

---

## 🧠 Modos de Inferência (Estratégias)

O sistema pode operar em 4 modos lógicos, que podem ser combinados com a infraestrutura de Threading.

### Mode 1: `CONTINUOUS` (Força Bruta)

* **Como funciona:** Processa cada quadro disponível, o mais rápido possível.
* **Comportamento:** Aquece rapidamente a CPU/NPU. Sofre de *Thermal Throttling* (perda de desempenho por calor) após alguns segundos.
* **Uso:** Apenas para benchmark de estresse ("Pior Caso").

### Mode 2: `BLINK_FIX` (Piscada Rítmica)

* **Inspiração:** O ato de piscar para "limpar" a visão (descanso).
* **Como funciona:** Fica acordado por `1.5s` e dorme forçadamente por `0.2s`.
* **Vantagem:** O descanso forçado permite que o hardware esfrie, mantendo a latência individual baixa (aprox. 90ms).
* **Desvantagem:** É "cego". Pode piscar bem na hora que alguém passa.

### Mode 3: `TIMELESS_BLINK` (Motion Gate)

* **Inspiração:** Visão Periférica Humana.
* **Como funciona:** Usa o `FrameGate`. Se a imagem está estática, o sistema dorme. Se há movimento, ele processa.
* **Vantagem:** Economia máxima de energia e dados. Gera CSVs pequenos.
* **Desvantagem:** Se houver movimento constante, vira um "Continuous".

### Mode 4: `HYBRID_SENTINEL` 

1. **Estado Calmo:** Usa `FrameGate`. Se nada acontece, dorme.
2. **Gatilho:** Se `FrameGate` detecta movimento -> Ativa YOLO.
3. **Adrenalina:** Se YOLO detecta `Pessoa` -> Ativa **Cooldown** (30 frames).
4. **Estado Alerta:** Durante o Cooldown, o sistema ignora o Gate e roda em modo Contínuo (Max FPS) para garantir a gravação do evento.

---

## ⚡ Infraestrutura: Threading

Esta opção (`use_threading=True`) pode ser ativada em qualquer modo.

* **STANDARD (False):** Leitura síncrona. O código para, espera a câmera capturar, lê, processa. Causa acúmulo de buffer (lag visual).
* **THREADED (True):** Leitura paralela. A câmera nunca para de capturar. O processamento sempre pega a imagem "do agora". Elimina a necessidade de limpeza de buffer (`cap.grab()`).

---

## 📊 Dicionário de Dados (`benchmark.csv`)

O arquivo CSV gerado contém as seguintes colunas para análise:

| Coluna | Descrição |
| --- | --- |
| `frame` | Número sequencial do quadro processado. |
| `infer_ms` | Tempo (ms) que a IA levou para processar a imagem. **0** significa que o sistema estava dormindo (Sleep). |
| `has_person` | `True` se a YOLO detectou uma pessoa (classe 0). |
| `num_objects` | Quantidade total de objetos detectados. |
| `mode` | Nome do modo lógico (ex: `HYBRID_SENTINEL`). |
| `infra` | Tipo de leitura (`THREADED` ou `STANDARD`). |
| `display` | `True` se a janela de vídeo estava sendo exibida (impacta performance). |
| `hw_fps` | **FPS Potencial ("Justiça"):** Cálculo de quantos quadros o hardware conseguiria fazer se não houvesse *sleep*. (`1000 / infer_ms`). |
| `timestamp` | Momento exato da gravação. |

---

## 🧪 Como Interpretar os Gráficos

1. **Estabilidade (Boxplot):** Procure por caixas "achatadas" e baixas. Caixas altas indicam instabilidade.
2. **Térmico (Lineplot):** Linhas que sobem indicam aquecimento (*throttling*). Linhas retas indicam saúde do hardware.
3. **Potencial (Barplot):** Mostra a força bruta. O modo `THREADED` deve apresentar as barras mais altas aqui.