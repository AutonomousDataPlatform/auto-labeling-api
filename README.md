# auto-labeling-api
auto-labeling-api for data extraction from Hanyang University

## 1. Settings
Two virtual environments are required.
- bigdata
- bigdata_clrnet

###
    cd fastapi-model-serving

### bigdata

Create virtual environment.

    conda create -n bigdata python=3.10

Install torch and libraries.

    conda activate bigdata
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
    pip install uvicorn[standard]
    pip install fastapi[all]
    pip install streamlit
    pip install ultralytics
    pip install --upgrade efficientnet-pytorch
    pip install -U requests_toolbelt
    pip install "numpy<2"

### bigdata_clrnet

Create virtual environment.

    conda create -n bigdata_clrnet python=3.10

Install torch and libraries.

    cd LaneDetection
    conda activate bigdata_clrnet
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
    pip install uvicorn[standard]
    pip install fastapi[all]
    pip install -r requirements.txt
    python setup.py build develop
    pip uninstall -y shapely
    pip install --no-cache-dir "shapely>=2.0.2"
    pip install "numpy<2"

### weights

Download weights file from the link below

    https://drive.google.com/file/d/1Ey6WLbdzGrsOzyuvhFaFV653Tjxe1Tn6/view?usp=sharing

Extract file

    unzip weights.zip

You will find the following two folders after extraction
    
    fastapi
      └─ weights
      └─ yolo11x_finetuned_kitti.pt
    LaneDetection
      └─ weights
Then, move folders (name: weights) below to the specified directions

    fastapi                         -->    auto-labeling-api
      └─ weights                            ├─ fastapi-model-serving
      └─ yolo11x_finetuned_kitti.pt             ├─ fastapi
                                                    ├─ weights
                                                    └─ yolo11x_finetuned_kitti.pt
    
    LaneDetection                   -->    auto-labeling-api
      └─ weights                            ├─ fastapi-model-serving
                                                ├─ LaneDetection
                                                    ├─ weights


## 2. Run

Terminal 1

    conda activate bigdata
    cd fastapi
    uvicorn server:app --host 0.0.0.0 --port 8000

Terminal 2

    conda activate bigdata_clrnet
    cd LaneDetection
    uvicorn server:app --host 0.0.0.0 --port 8001

Terminal 3

    conda activate bigdata
    cd streamlit
    streamlit run ui.py
