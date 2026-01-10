from ultralytics import YOLO

# Carrega o modelo com os pesos pré-treinados (nano version)
model = YOLO("yolov8n.pt")

# Inicia a inferência na webcam (source=0).
# stream=True: Cria um gerador para processar vídeo de forma eficiente (memória).
# classes=[0]: Filtra para detectar apenas a classe "pessoa".
results = model.predict(0, stream=True, classes=[0], show=True)

for result in results:
    boxes = result.boxes

    if boxes:
        # Itera sobre cada caixa delimitadora detectada no frame atual
        for box in boxes.xyxy:
            # box contém as coordenadas brutas (Tensores)
            print(f"Pessoa detectada em: ", {box})