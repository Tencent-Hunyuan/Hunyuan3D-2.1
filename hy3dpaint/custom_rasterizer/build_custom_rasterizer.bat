@echo off
echo Building custom rasterizer for Windows...
python setup.py build_ext --inplace
if %ERRORLEVEL% neq 0 (
    echo Build failed! Please check error messages above.
    exit /b 1
)
echo Custom rasterizer build completed successfully.
cd ..\.. 