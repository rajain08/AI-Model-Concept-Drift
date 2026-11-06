FROM python:3.12-slim

WORKDIR /app
COPY scripts/cloud_prototype.py /app/scripts/cloud_prototype.py
COPY scripts/run_cloud_pipeline.py /app/scripts/run_cloud_pipeline.py
RUN mkdir -p /app/results

ENV PORT=8080
ENV PROTOTYPE_DB=/app/results/cloud_prototype.db
EXPOSE 8080

CMD ["python", "scripts/cloud_prototype.py"]
