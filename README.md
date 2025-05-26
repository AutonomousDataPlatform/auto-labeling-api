# auto-labeling-api
auto-labeling-api for data extraction from Hanyang University

Two virtual environments are required.
- bigdata
- bigdata_clrnet

### bigdata

Create virtual environment.

    conda create -n bigdata python=3.10

Install torch and libraries.

    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics
    pip install --upgrade efficientnet-pytorch
    pip install -U requests_toolbelt

### bigdata_clrnet

Create virtual environment.

    conda create -n bigdata_clrnet python=3.10

Install torch and libraries.

    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
    cd LaneDetection
    pip install -r requirements.txt
    python setup.py build develop
    pip uninstall -y shapely
    pip install --no-cache-dir "shapely>=2.0.2"

