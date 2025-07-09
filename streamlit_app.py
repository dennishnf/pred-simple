import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Cargar el modelo
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('modelo-pesos.pth', map_location=torch.device('cpu')))

# Modelo en modo evaluación
model.eval()

# Definir las transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para hacer predicciones
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

        if predicted.item() == 0: return "GATO"
        if predicted.item() == 1: return "PERRO"

# Interfaz de Streamlit
st.title('Clasificador de gatos y perros')
uploaded_file = st.file_uploader("Elija una imagen...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', width=400)
    st.write("Clasificando...")
    label = predict(image)
    st.write(f"Predicción: {label}")
    
