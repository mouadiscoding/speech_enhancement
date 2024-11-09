FROM python:3.11.3

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./speech/Lib/site-packages/pysepm_evo/qualityMeasures.py /usr/local/lib/python3.11/site-packages/pysepm_evo/qualityMeasures.py

COPY ./speech/Lib/site-packages/pysepm_evo/util.py /usr/local/lib/python3.11/site-packages/pysepm_evo/util.py

COPY ./app /code/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]