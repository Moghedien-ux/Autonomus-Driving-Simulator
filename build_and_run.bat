@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set VK_SDK=C:\VulkanSDK\1.4.363.0
echo Compiling Autonomous Driving Simulator with Vulkan 1.4 + EngineZero Core (Release /O2)...
cl /nologo /O2 /std:c++20 /EHsc /W3 /I"%VK_SDK%\Include" main.cpp engine_zero\Tensor.cpp engine_zero\Autograd.cpp engine_zero\Ops.cpp "%VK_SDK%\Lib\vulkan-1.lib" user32.lib gdi32.lib /Fe:autonomy_sim.exe
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)
echo Build succeeded: autonomy_sim.exe
if "%1"=="--headless-test" (
    autonomy_sim.exe --headless-test
) else if "%1"=="--smoke-test-gui" (
    autonomy_sim.exe --smoke-test-gui
) else if "%1"=="--build-only" (
    echo Done.
) else (
    start "" autonomy_sim.exe
)
