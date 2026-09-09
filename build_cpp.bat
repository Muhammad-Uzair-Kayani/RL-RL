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
for /f "tokens=*" %%a in ('%PY% -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"') do set "PY_LIBDIR=%%a"

if not exist "%PY_LIBDIR%\python*.lib" (
    for /f "tokens=*" %%a in ('%PY% -c "import os, sys; print(os.path.dirname(sys.executable))"') do set "PY_BASE=%%a"
    set "PY_LIBDIR=%PY_BASE%\libs"
)

mkdir "%ROOT%python" 2>nul

cl.exe /std:c++17 /O2 /MD /EHsc /I "%PY_INC%" /I "%PYBIND_INC%" /LD cpp\rl_env.cpp /link /LIBPATH:"%PY_LIBDIR%" /OUT:python\teamsports_rl.pyd

echo.
if exist "%ROOT%python\teamsports_rl.pyd" (
    echo Build succeeded: %ROOT%python\teamsports_rl.pyd
) else (
    echo Build failed.
)
