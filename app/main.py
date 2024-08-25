from io import BytesIO  # Import BytesIO
from time import time

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from PIL import Image
from prometheus_client import start_http_server
from torchvision import transforms

from model import MyModel

# Load the model
model = MyModel()
model.load_state_dict(torch.load("../model/model_10.pth"))
# model = torch.load("../model/model_10.pth")
model.eval()  # Set the model to evaluation mode
logger.info("Model loaded successfully")

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logger.info(f"Using {device}")

app = FastAPI()

start_http_server(port=8099, addr="0.0.0.0")
resource = Resource(attributes={SERVICE_NAME: "app-service"})
reader = PrometheusMetricReader()
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)
meter = metrics.get_meter("myapp", "0.0.1")
counter = meter.create_counter(
    name="request_counter", description="Number of app requests"
)

histogram = meter.create_histogram(
    name="response_histogram",
    description="app response histogram",
    unit="seconds",
)

# Define the image transformations (resize, normalize, convert to tensor)
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])


@app.post("/query")
async def query(image_file: UploadFile = File(...)):
    starting_time = time()

    # Read the image file
    request_image_content = await image_file.read()

    # Load the image content with PIL
    image = Image.open(BytesIO(request_image_content)).convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():  # Disable gradient computation
        output = model(input_tensor)

    # Process the model output (e.g., get the predicted class or probabilities)
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # Labels for all metrics
    label = {"api": "/query"}

    # Increase the counter
    counter.add(10, label)

    # Mark the end of the response
    ending_time = time()
    elapsed_time = ending_time - starting_time

    # Add histogram
    logger.info("elapsed time: ", elapsed_time)
    logger.info(elapsed_time)
    histogram.record(elapsed_time, label)

    return {"predicted_class": predicted_class}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
# uvicorn main:app --host 0.0.0.0 --port 8088
