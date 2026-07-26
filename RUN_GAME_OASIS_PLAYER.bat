@echo off
cd /d "%~dp0"
py -3 roblox_action_flow_app.py
if errorlevel 1 (
  echo.
  echo The player could not start. Install the required packages with:
  echo py -3 -m pip install -r requirements.txt
  pause
)
