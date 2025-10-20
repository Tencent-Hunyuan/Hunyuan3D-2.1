@echo off
setlocal EnableDelayedExpansion

echo Compiling mesh_inpaint_processor...

REM Check compiler option
if "%1"=="mingw" (
    goto :USE_MINGW
) else (
    goto :USE_MSVC
)

:USE_MSVC
echo Using MSVC compiler...

REM Get pybind11 includes
for /f "tokens=*" %%i in ('python -m pybind11 --includes') do set PYBIND_INCLUDES=%%i

REM Get Python extension suffix
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set PY_SUFFIX=%%i

REM Get Python library directory
for /f "tokens=*" %%i in ('python -c "import sys; import os; print(os.path.join(sys.base_prefix, 'libs'))"') do set PYTHON_LIB_DIR=%%i

REM Check if Python libs directory exists
if not exist "!PYTHON_LIB_DIR!" (
    echo Warning: Python library directory not found at !PYTHON_LIB_DIR!
    
    REM Try another common location for Python libs
    for /f "tokens=*" %%i in ('python -c "import sys; import os; print(os.path.join(os.path.dirname(sys.executable), 'libs'))"') do set PYTHON_LIB_DIR=%%i
    
    if not exist "!PYTHON_LIB_DIR!" (
        echo Error: Cannot find Python library directory
        echo Please ensure python310.lib is available in your Python installation
        exit /b 1
    )
)

echo Found Python libraries at: !PYTHON_LIB_DIR!

REM Check if Visual Studio environment is already set
where cl >nul 2>nul
if %ERRORLEVEL% equ 0 (
    goto :DO_MSVC_COMPILE
)

REM Try to find and activate Visual Studio environment automatically
set FOUND_VS=0

REM Try VS2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022, activating environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    set FOUND_VS=1
    goto :DO_MSVC_COMPILE
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022, activating environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
    set FOUND_VS=1
    goto :DO_MSVC_COMPILE
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    echo Found VS2022, activating environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
    set FOUND_VS=1
    goto :DO_MSVC_COMPILE
)

REM Try VS2019
if %FOUND_VS% equ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        echo Found VS2019, activating environment...
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
        set FOUND_VS=1
        goto :DO_MSVC_COMPILE
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
        echo Found VS2019, activating environment...
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
        set FOUND_VS=1
        goto :DO_MSVC_COMPILE
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
        echo Found VS2019, activating environment...
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
        set FOUND_VS=1
        goto :DO_MSVC_COMPILE
    )
)

REM Check if VS was found
where cl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Could not find cl compiler. Please run Visual Studio's vcvarsall.bat first or use mingw option.
    echo Tips: Try one of these commands:
    echo       compile_mesh_painter.bat mingw
    echo       or first run "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    echo       (Visual Studio path may be different on your system)
    exit /b 1
)

:DO_MSVC_COMPILE
echo Compiling with MSVC...
REM Compile with MSVC
cl /O2 /W4 /std:c++14 /LD %PYBIND_INCLUDES% mesh_inpaint_processor.cpp /link /LIBPATH:"!PYTHON_LIB_DIR!" /out:mesh_inpaint_processor%PY_SUFFIX%
if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    exit /b 1
)
goto :END

:USE_MINGW
echo Using MinGW-G++ compiler...

REM Check if MinGW is installed
where g++ >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: g++ compiler not found. Please install MinGW or use Visual Studio.
    exit /b 1
)

REM Get pybind11 includes
for /f "tokens=*" %%i in ('python -m pybind11 --includes') do set PYBIND_INCLUDES=%%i

REM Get Python extension suffix
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set PY_SUFFIX=%%i

REM Get Python library directory and filename for MinGW
for /f "tokens=*" %%i in ('python -c "import sys; from sysconfig import get_config_var; import os; print(os.path.join(sys.base_prefix, 'libs', f'python{get_config_var(\"py_version_nodot\")}.dll'))"') do set PYTHON_LIB=%%i

REM Compile with MinGW
g++ -O3 -Wall -shared -std=c++11 %PYBIND_INCLUDES% mesh_inpaint_processor.cpp -L"!PYTHON_LIB!" -o mesh_inpaint_processor%PY_SUFFIX%
if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    exit /b 1
)

:END
echo Compilation completed!
echo Generated module: mesh_inpaint_processor%PY_SUFFIX% 