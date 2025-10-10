Holistic Horizon EPM - Setup and Run

This project contains Phase 1-4 scripts for building an emissions-intensity
baseline and hybrid models. This README shows how to create a Python virtual
environment and install the required packages (including TensorFlow).

Important notes
- TensorFlow (large package) is required for LSTM training and for Phases 3/4.
- If you run into errors installing TensorFlow on your Python version, consider
  using Python 3.10 or 3.11, or installing via Conda where prebuilt binaries
  may be easier to obtain.

1) Create and activate a virtual environment (Windows cmd.exe)

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install dependencies from requirements.txt

```cmd
python -m pip install -r requirements.txt
```

Note: installing `tensorflow` may take a few minutes and may print many lines
while compiling/wheel-installing. If you prefer to skip TensorFlow and run the
baseline with a RandomForest fallback, remove or comment out `tensorflow` from
`requirements.txt` and run Phase 1 with the `--force-rf` flag.

3) Run Phase 1 (example)

```cmd
python "Phase 1.py" --file "C:\Users\acer\Downloads\Enterprise_Sustainable Power Evaluation_Dataset.csv" --epochs 20 --save-artifacts --save-dir "artifacts"
```

4) Troubleshooting
- If you see errors like "No matching distribution found for tensorflow" or
  version incompatibility errors, try creating a Conda environment with a
  compatible Python version (3.10/3.11) and install TensorFlow there.
- To run quickly without TF, use the RandomForest fallback:

```cmd
python "Phase 1.py" --force-rf --save-artifacts
```

Files created
- `requirements.txt` - the package list to install
- `README.md` - this file with setup/run instructions

If you'd like, I can also: create a small script to create & activate the venv,
add a `Makefile`/`tasks.json` for VS Code, or pin exact package versions that I
validated in my environment. Which would you prefer next?