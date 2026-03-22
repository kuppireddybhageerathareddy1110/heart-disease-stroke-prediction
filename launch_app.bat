@echo off
REM Heart Disease Prediction App - Quick Launch Script
REM ===================================================

echo.
echo ========================================
echo   Heart Disease Prediction App
echo   Launching Streamlit Application...
echo ========================================
echo.

REM Check if models exist
if not exist "models\best_model.pkl" (
    echo ERROR: Model files not found!
    echo.
    echo Please run the pipeline first:
    echo   python heart_disease_pipeline_improved.py
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not installed!
    echo.
    echo Installing requirements...
    pip install -r requirements.txt
    echo.
)

echo Starting Streamlit app...
echo.
echo The app will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run streamlit_app.py

pause
