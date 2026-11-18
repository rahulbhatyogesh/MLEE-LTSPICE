set -e
PY="/c/Users/rahul/AppData/Local/Programs/Python/Python312/python.exe"
"$PY" -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install ultralytics opencv-python scikit-image python-docx huggingface_hub pytest matplotlib pandas scikit-learn
echo "SETUP_DONE"
