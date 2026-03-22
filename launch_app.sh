#!/bin/bash
# Heart Disease Prediction App - Quick Launch Script
# ===================================================

echo ""
echo "========================================"
echo "  Heart Disease Prediction App"
echo "  Launching Streamlit Application..."
echo "========================================"
echo ""

# Check if models exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "ERROR: Model files not found!"
    echo ""
    echo "Please run the pipeline first:"
    echo "  python heart_disease_pipeline_improved.py"
    echo ""
    exit 1
fi

# Check if streamlit is installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Streamlit not installed!"
    echo ""
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo ""
fi

echo "Starting Streamlit app..."
echo ""
echo "The app will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
