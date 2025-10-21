# =========================
# Stage 1: Model Training
# =========================
FROM python:3.9 AS model_training

WORKDIR /app

# Copy training script and dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/model_training.py /app/

# Train and persist artifacts
RUN python model_training.py


# =========================
# Stage 2: Serve Predictions
# =========================
FROM python:3.9 AS serving

WORKDIR /app

# Copy trained artifacts from stage 1
COPY --from=model_training /app/cancer_model.keras /app/
COPY --from=model_training /app/scaler.pkl /app/
COPY --from=model_training /app/metadata.json /app/

# App source
COPY requirements.txt /app/
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

COPY src/main.py /app/
COPY src/templates /app/templates
COPY src/statics /app/statics

# Runtime config
ENV FLASK_ENV=production
EXPOSE 4000

# Start the Flask server
CMD ["python", "main.py"]
