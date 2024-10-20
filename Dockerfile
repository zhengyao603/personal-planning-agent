FROM python:3.11-slim

WORKDIR /personal-agent
COPY . /personal-agent

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
CMD ["agent.py1"]