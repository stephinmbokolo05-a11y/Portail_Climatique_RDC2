@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d D:\Portail_Climatique_RDC\ATLAS
set PY=C:\Users\hp\anaconda3\envs\env_atlas\python.exe
if not exist "%PY%" set PY=python
echo.
echo === Generation de l'atlas pour GitHub ===
echo.
"%PY%" build_hf_space.py
echo.
echo Si vous voyez le mot TERMINE ci-dessus, le dossier pour_github est pret.
echo Sinon, faites une capture de cette fenetre et envoyez-la.
echo.
pause
