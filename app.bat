@echo off


REM Check if the virtual environment folder exists
if not exist "venv" (
    echo Setting up virtual environment...
    python -m venv venv
    echo Activating virtual environment...
    call venv\Scripts\activate.bat || exit /b
    echo Installing requirements...
    pip install -r requirements.txt || exit /b
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Run the app.py file
echo Running app.py...
call venv\Scripts\python.exe -m shiny run --port 50135 --reload --autoreload-port 50136 app.py

REM Prevent the command prompt from closing
pause