@echo off
title UrbanGNN Pipeline Runner
echo ===============================================
echo      Urban GNN Full Pipeline (Training + XE)
echo ===============================================
echo.

REM ---- Activate venv if exists ----
if exist "%~dp0venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo No venv found. Using system Python.
)
echo.

REM ---- Clean old outputs ----
echo Cleaning old outputs...
del "gnn_prediction.csv" >nul 2>&1
del "urban_gnn_model.pth" >nul 2>&1
del "feature_importance.csv" >nul 2>&1
del "ablation_study.csv" >nul 2>&1
del "gnn_explain_edges_node*.csv" >nul 2>&1
del "gnn_residuals.csv" >nul 2>&1
del "plot_*.png" >nul 2>&1
del "graph_*.png" >nul 2>&1

REM --- NEW: Delete old SHAP data to force update ---
del "*.npy" >nul 2>&1
REM -------------------------------------------------

echo Done.
echo.

REM ---- Training ----
echo [1/4] Training model (train_gnn.py)...
python train_gnn.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR during training!
    pause
    exit /b 1
)
echo Training Completed.
echo.

REM ---- Explainability ----
echo [2/4] Running explainability (explainability.py)...
python explainability.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in explainability!
    pause
    exit /b 1
)
echo Explainability Completed.
echo.

REM ---- Plotting ----
echo [3/4] Generating plots (plotting.py)...
python plotting.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in plotting!
    pause
    exit /b 1
)
echo Plotting Completed.
echo.

REM ---- Graph Visualization ----
echo [4/4] Visualizing Graph Structure (visualize_graph.py)...
python visualize_graph.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in graph visualization!
    pause
    exit /b 1
)
echo Graph Visualization Completed.
echo.

REM ---- Open folder ----
echo Opening output folder...
explorer "%cd%"

echo ===============================================
echo       Full GNN Pipeline Completed!
echo   - gnn_prediction.csv
echo   - urban_gnn_model.pth
echo   - feature_importance.csv
echo   - ablation_study.csv
echo   - explainer edges
echo   - plot images
echo   - graph_full.png / graph_subgraph_500.png
echo ===============================================
echo.
pause