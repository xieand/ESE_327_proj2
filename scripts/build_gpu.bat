@echo off
rem Builds the CUDA programs (bin\<name>_gpu.exe) on Windows.
rem
rem nvcc drives the MSVC compiler, so this script first loads the Visual
rem Studio x64 environment (unless cl.exe is already on the PATH) and then
rem does what "make gpu" does on Linux.
rem
rem usage: scripts\build_gpu.bat [name ...]      (default: dtree nbayes knn logreg)
setlocal enabledelayedexpansion
cd /d "%~dp0.."

where nvcc >nul 2>nul
if errorlevel 1 (
    echo error: nvcc is not on the PATH ^(install the CUDA Toolkit^)
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo error: cl.exe is not on the PATH and vswhere.exe was not found
        exit /b 1
    )
    set "VSDIR="
    for /f "usebackq delims=" %%i in (`"!VSWHERE!" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
    if not defined VSDIR (
        echo error: no Visual Studio installation with the C++ tools was found
        exit /b 1
    )
    call "!VSDIR!\VC\Auxiliary\Build\vcvars64.bat" >nul
    if errorlevel 1 exit /b 1
)

if not defined CUDA_ARCH set "CUDA_ARCH=-arch=native"
rem --fmad=false       no fused multiply-add: the kernels round like the CPU code
rem /Zc:preprocessor   the CUB headers need the standard-conforming preprocessor
set NVCCFLAGS=-O3 -std=c++17 %CUDA_ARCH% --fmad=false -Xcompiler /Zc:preprocessor -Xcompiler /EHsc -Xcompiler /D_CRT_SECURE_NO_WARNINGS
set COMMON=src\dataset.cpp src\eval.cpp src\runner.cpp src\util.cpp
set dtree_SRC=src\dtree.cpp
set nbayes_SRC=
set knn_SRC=src\encode.cpp
set logreg_SRC=src\encode.cpp

set NAMES=%*
if "%NAMES%"=="" set NAMES=dtree nbayes knn logreg

if not exist bin mkdir bin
for %%n in (%NAMES%) do (
    echo nvcc %%n_gpu
    nvcc %NVCCFLAGS% -Iinclude src\%%n_gpu.cu !%%n_SRC! %COMMON% -o bin\%%n_gpu.exe
    if errorlevel 1 exit /b 1
)
echo done
