@echo off
title UrbanGNN Pipeline Runner
echo ===============================================
echo      Urban GNN Full Pipeline (Training + XE)
echo      Input: fishnet_final_composite.shp
echo      Features: 15  ^|  Targets: ECSI, SEVI
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
del /q "gnn_prediction.csv" 2>nul
del /q "urban_gnn_model.pth" 2>nul
del /q "feature_importance.csv" 2>nul
del /q "ablation_study.csv" 2>nul
del /q "grouped_ablation.csv" 2>nul
del /q "pdp_results.csv" 2>nul
del /q "gnn_explain_edges_node*.csv" 2>nul
del /q "baseline_comparison.csv" 2>nul
del /q "learning_curve.png" 2>nul
del /q "vif_report.csv" 2>nul
del /q "*.npy" 2>nul
del /q "shap_feature_names.csv" 2>nul
del /q "plot_*.png" 2>nul
del /q "graph_*.png" 2>nul
del /q "shap_*.png" 2>nul
echo Done.
echo.

REM ---- Step 1: Training ----
echo [1/5] Training GATv2 model (train_gnn.py)...
python train_gnn.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in train_gnn.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Training complete.
echo.

REM ---- Step 2: Archetype classification ----
echo [2/6] Urban archetype classification (compute_archetypes.py)...
python compute_archetypes.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in compute_archetypes.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Archetype classification complete.
echo.

REM ---- Step 3: Moran's I ----
echo [3/6] Spatial autocorrelation check (morans_i_analysis.py)...
python morans_i_analysis.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in morans_i_analysis.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Moran's I complete.
echo.

REM ---- Step 3: Explainability ----
echo [4/6] Running explainability (explainability.py)...
python explainability.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in explainability.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Explainability complete.
echo.

REM ---- Step 4: Plotting ----
echo [5/6] Generating SHAP plots (plotting.py)...
python plotting.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in plotting.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Plotting complete.
echo.

REM ---- Step 5: Graph visualisation ----
echo [6/6] Graph visualisation (visualize_graph.py)...
python visualize_graph.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR in visualize_graph.py -- pipeline stopped.
    pause
    exit /b 1
)
echo Graph visualisation complete.
echo.

REM ---- Open output folder ----
echo Opening output folder...
explorer "%cd%"

echo.
echo ===============================================
echo       Full GNN Pipeline Completed!
echo.
echo   PRIMARY outputs (cite in paper):
echo     archetypes.csv
echo     archetype_summary.csv
echo     silhouette_summary.csv
echo     gnn_prediction.csv
echo     baseline_comparison.csv
echo     grouped_ablation.csv
echo     pdp_results.csv
echo.
echo   SUPPLEMENTARY:
echo     ablation_study.csv
echo     feature_importance.csv
echo     shap_grouped_importance.csv
echo     morans_scatter_residuals.png
echo     shap_beeswarm_comparison.png
echo     shap_dep_*.png
echo     graph_structure_ECSI/SEVI.png
echo ===============================================
echo.
pause
