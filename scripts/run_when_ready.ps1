# Waits for the Cholec80 subset download to finish, then extracts frames, trains and evaluates.
# Progress goes to train.log. Safe to re-run: finished stages are skipped.
# Run from anywhere; paths are resolved relative to this script. Set $env:PYTHON to choose an interpreter.
$proj = Split-Path -Parent $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
Set-Location $proj
while ($true) {
  $done = $false
  try { $done = (Get-Content "$proj\data\download_status.json" -Raw | ConvertFrom-Json).finished } catch {}
  if ($done) { break }
  Start-Sleep 60
}
Add-Content "$proj\train.log" "$(Get-Date -Format 'HH:mm:ss') download complete; starting frame extraction and training"
for ($try = 1; $try -le 3; $try++) {
  & cmd.exe /c "`"$py`" -u -m surgphase.train >> train_console.log 2>&1"
  if (Select-String -Path "$proj\train.log" -Pattern 'ALL DONE' -Quiet) { break }
  Add-Content "$proj\train.log" "$(Get-Date -Format 'HH:mm:ss') training stopped early (attempt $try); restarting in 60 s"
  Start-Sleep 60
}
