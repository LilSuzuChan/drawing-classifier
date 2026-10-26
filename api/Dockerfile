FROM python:3.12-slim
WORKDIR /app
COPY ml/src/drawing_classifier/feedback.py /app/drawing_classifier/feedback.py
COPY api/server.py /app/server.py
COPY web/public/models/mnist_model.onnx /app/bundled.onnx
USER 1000:1000
EXPOSE 8080
CMD ["python", "server.py"]
