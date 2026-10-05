@echo off
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if errorlevel 1 (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if errorlevel 1 (
    echo Could not find Visual Studio 2022 vcvars64.bat
    exit /b 1
)

set "ROOT=%~dp0"
set "PY=venv\Scripts\python.exe"
set "PYBIND_INC=%ROOT%venv\Lib\site-packages\pybind11\include"

for /f "tokens=*" %%a in ('%PY% -c "import sysconfig; print(sysconfig.get_path('include'))"') do set "PY_INC=%%a"

for /f "tokens=*" %%a in ('%PY% -c "import os, sys; exe = sys.executable; base = os.path.dirname(os.path.dirname(exe)); print(os.path.join(base, 'libs'))"') do set "PY_LIBDIR=%%a"

mkdir "%ROOT%python" 2>nul

cl.exe /std:c++14 /O2 /MD /EHsc /I "%PY_INC%" /I "%PYBIND_INC%" /LD cpp\rl_env.cpp /link /LIBPATH:"%PY_LIBDIR%" /OUT:python\teamsports_rl.pyd

echo.
if exist "%ROOT%python\teamsports_rl.pyd" (
    echo Build succeeded: %ROOT%python\teamsports_rl.pyd
) else (
    echo Build failed.
)
