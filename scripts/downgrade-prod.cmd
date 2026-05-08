@echo off
setlocal EnableDelayedExpansion
rem Downgrade PROD (live trading) onemil node back to t3.large baseline.
rem Pass /y to skip the confirmation prompt.
rem WARNING: prod runs the live trader. Do NOT resize during US market hours
rem          (09:30-16:00 ET) without an explicit reason.

set REGION=eu-north-1
set TARGET_TYPE=t3.large
set INSTANCE_ID=i-0601c2960e37bf5f4

echo.
echo === Current state ===
aws ec2 describe-instances --region %REGION% --instance-ids %INSTANCE_ID% --query "Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]" --output table

echo.
echo Plan: PROD (%INSTANCE_ID%) -^> %TARGET_TYPE%
echo Estimated downtime: 3-5 min
echo WARNING: this is the LIVE trading node.

if /i "%~1"=="/y" goto :proceed
set /p CONFIRM=Proceed with PROD downgrade? (y/N):
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
echo Downgrade complete. SSM tunneling typically takes 60-90s to come online.
echo Smoke test: ssh prod-onemil-claude "uname -a; nproc; free -h; systemctl is-active onemil-trader"
endlocal
