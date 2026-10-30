FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHRONOFLUX_NO_BROWSER=1 CHRONOFLUX_USERS_DB=/app/data/state/users.db CHRONOFLUX_SECRET_KEY_PATH=/app/data/state/secret_key
EXPOSE 8000
CMD ["python", "run.py"]
