FROM python:3.11.1-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY src/ .

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]