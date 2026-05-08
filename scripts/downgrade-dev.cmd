@echo off
setlocal EnableDelayedExpansion
rem Downgrade dev (paper) onemil node back to t3.large baseline.
rem Pass /y to skip the confirmation prompt.

set REGION=eu-north-1
set TARGET_TYPE=t3.large
set DEV_EIP=13.61.40.28

echo Resolving dev instance ID via EIP %DEV_EIP%...
for /f "tokens=*" %%i in ('aws ec2 describe-addresses --region %REGION% --public-ips %DEV_EIP% --query "Addresses[0].InstanceId" --output text') do set INSTANCE_ID=%%i
if "%INSTANCE_ID%"=="" (
    echo ERROR: could not resolve dev instance ID
    exit /b 1
)

echo.
echo === Current state ===
aws ec2 describe-instances --region %REGION% --instance-ids %INSTANCE_ID% --query "Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]" --output table

echo.
echo Plan: dev (%INSTANCE_ID%) -^> %TARGET_TYPE%
echo Estimated downtime: 3-5 min

if /i "%~1"=="/y" goto :proceed
set /p CONFIRM=Proceed with downgrade? (y/N):
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    exit /b 1
)

:proceed
echo.
echo [1/3] Stopping...
aws ec2 stop-instances --region %REGION% --instance-ids %INSTANCE_ID% >nul || exit /b 1
aws ec2 wait instance-stopped --region %REGION% --instance-ids %INSTANCE_ID% || exit /b 1

echo [2/3] Modifying instance type to %TARGET_TYPE%...
aws ec2 modify-instance-attribute --region %REGION% --instance-id %INSTANCE_ID% --instance-type Value=%TARGET_TYPE% || exit /b 1

echo [3/3] Starting...
aws ec2 start-instances --region %REGION% --instance-ids %INSTANCE_ID% >nul || exit /b 1
aws ec2 wait instance-running --region %REGION% --instance-ids %INSTANCE_ID% || exit /b 1

echo.
echo === After ===
aws ec2 describe-instances --region %REGION% --instance-ids %INSTANCE_ID% --query "Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]" --output table

echo.
echo Downgrade complete. Allow ~30-60s before SSH responds.
echo Smoke test: ssh dev-onemil-claude "uname -a; nproc; free -h; systemctl is-active onemil-trader"
endlocal
