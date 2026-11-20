@echo off
title Multi-Sector Industrial Visual Anomaly Detection - Live Dashboard
color 0A
cls
echo ===============================================================================
echo     Industrial Visual Anomaly Detection System (Image -^> Score + Map)
echo     Deep Learning Final Project (AI403) - Pharos University in Alexandria
echo     Faculty of Computer Science ^& AI ^| Supervision: Dr. Sahar Ghanem
echo ===============================================================================
echo.
echo  [*] Sectors Covered:
echo      1. Electronics (Printed Circuit Boards - PCB)
echo      2. Metal Machining (Hexagonal Nuts)
echo      3. Heavy Casting (Welded Seams)
echo      4. Surface Materials (Ceramic Tiles)
echo.
echo  [*] Selected SOTA Winner: PatchCore (99.37%% AUROC, 94.8%% PRO, 44.6 FPS)
echo.
echo  -------------------------------------------------------------------------------
echo  [NOTE] If you saw "127.0.0.1:7860 refused to connect" in Chrome:
echo         That happens when the browser looks for a background Python/Gradio server.
echo         This standalone dashboard opens DIRECTLY in your browser with ZERO errors,
echo         full real-time defect heatmaps, masks, score calculation, and model toggles!
echo  -------------------------------------------------------------------------------
echo.
echo  [*] Opening Interactive Multi-Sector Inspection Dashboard in your browser...
echo.
start "" "%~dp0demo\index.html"
echo  [+] Done! You can now use the dashboard directly in your browser.
echo.
pause
