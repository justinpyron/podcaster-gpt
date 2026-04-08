FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
# NOTE: requirements.txt is generated during the CI/CD workflow from pyproject.toml.
# This keeps dependencies in sync without manual duplication.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application files
COPY app.py ./

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run Streamlit with Cloud Run-compatible settings
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
