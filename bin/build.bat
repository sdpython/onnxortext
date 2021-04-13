@echo off
set current=%~dp0
set root=%current%..
cd %root%
@echo ##################
@echo Compile
@echo running %root%\setup.py build_ext --inplace
@echo ##################
set pythonexe="c:\Python387_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python372_x64\python.exe"
%pythonexe% -u %root%\setup.py build_ext --inplace
if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Compile.
@echo ##################
@echo Build
@echo running %root%\setup.py bdist_wheel
@echo ##################
%pythonexe% -u %root%\setup.py bdist_wheel sdist
if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Build.
cd %current%