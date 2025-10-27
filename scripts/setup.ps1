Param(
  [string]$Method = "venv"  # options: venv|conda
)

$ROOT = Split-Path -Parent $PSScriptRoot

if ($Method -eq "conda") {
  Write-Host "Creating/activating conda env 'agriprofit'..."
  conda env remove -n agriprofit -y | Out-Null
  conda env create -f "$ROOT\environment.yml"
  Write-Host "Run: conda activate agriprofit"
  Write-Host "Then: streamlit run ui/app_streamlit.py"
} else {
  Write-Host "Creating Python venv in .venv..."
  python -m venv "$ROOT\.venv"
  & "$ROOT\.venv\Scripts\Activate.ps1"
  python -m pip install --upgrade pip
  pip install -r "$ROOT\requirements.txt"
  Write-Host "Venv activated. Run: streamlit run ui/app_streamlit.py"
}

