@echo off
setlocal

echo Stopping SignalRank local demo...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$processes = Get-CimInstance Win32_Process;" ^
  "$roots = $processes | Where-Object { " ^
  "    $_.ProcessId -ne $PID -and $_.CommandLine -and (" ^
  "        $_.CommandLine -like '*signalrank.api.main:app*' -or " ^
  "        $_.CommandLine -like '*src\signalrank\ui\app.py*'" ^
  "    )" ^
  "};" ^
  "function Stop-ProcessTree([int]$procId) {" ^
  "    $processes | Where-Object { $_.ParentProcessId -eq $procId } | ForEach-Object {" ^
  "        Stop-ProcessTree $_.ProcessId" ^
  "    };" ^
  "    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue" ^
  "};" ^
  "$roots | Sort-Object ProcessId -Unique | ForEach-Object {" ^
  "    Stop-ProcessTree $_.ProcessId" ^
  "}"

timeout /t 1 /nobreak >nul

echo.
echo Checking demo ports...

powershell -NoProfile -Command ^
  "$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | " ^
  "Where-Object { $_.LocalPort -eq 8000 -or ($_.LocalPort -ge 8501 -and $_.LocalPort -le 8599) };" ^
  "if ($listeners) {" ^
  "    Write-Host 'Remaining listeners:';" ^
  "    $listeners | Select-Object LocalAddress,LocalPort,OwningProcess | Format-Table -AutoSize" ^
  "} else {" ^
  "    Write-Host 'SignalRank demo ports are clear.'" ^
  "}"

echo.
echo Done.