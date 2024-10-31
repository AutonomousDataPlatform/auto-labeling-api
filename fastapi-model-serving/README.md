# fastapi-model-serving

Reference: https://github.com/davidefiocco/streamlit-fastapi-model-serving

Requrirements
- docker-compose

To run the example in a machine running Docker and docker compose, run:

    docker-compose up --build -d

To visit the FastAPI documentation of the resulting service, visit http://localhost:8000/docs with a web browser.  
To visit the streamlit UI, visit http://localhost:8501.

Logs can be inspected via:

    docker-compose logs

### Debugging

To modify and debug the app, [development in containers](https://davidefiocco.github.io/debugging-containers-with-vs-code) can be useful (and kind of fun!).

### Run manual

conda activate bigdata
cd fastapi
uvicorn server:app --host 0.0.0.0 --port 8000

conda activate bigdata_clrnet
cd LaneDetection
uvicorn server:app --host 0.0.0.0 --port 8001

conda activate bigdata
cd streamlit
streamlit run ui.py