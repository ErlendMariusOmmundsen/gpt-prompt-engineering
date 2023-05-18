#!/bin/bash
 
set -e

# This script creates a custom conda environment and kernel based on a sample yml file.

conda env create -f env.yml
echo "Activating new conda environment"
conda activate thoth

PACKAGE=numpy 
ENVIRONMENT=thoth
conda activate "$ENVIRONMENT"
pip install torch-tb-profiler~=0.4.0 numpy tiktoken azureml-mlflow==1.50.0 rouge-score inference-schema~=1.5.0 bert-score debugpy~=1.6.3 azureml-dataset-runtime==1.50.0 azureml-telemetry==1.50.0 azureml-core==1.50.0 markupsafe==2.1.2 azureml-contrib-services==1.50.0 azure-ml==0.0.1 azureml-inference-server-http~=0.8.0 azure-ml-component==0.9.18 py-spy==0.3.12 language-tool-python ipykernel~=6.20.2 azureml-defaults==1.50.0
echo "Installing kernel"
sudo -u caleb -i <<'EOF'
conda activate thoth
python -m ipykernel install --user --name thoth --display-name "thoth_kernel"
echo "Conda environment setup successfully."
EOF
