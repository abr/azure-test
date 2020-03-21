wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --quiet -O miniconda.sh
bash miniconda.sh -b -p ./miniconda
export PATH="$PWD/miniconda/bin:$PATH"
conda create -y -n azure-test python=3.6
source activate azure-test
pip install nengo==3.0.0 nengo-dl==3.1.0 azure-iot-device==2.1.0
