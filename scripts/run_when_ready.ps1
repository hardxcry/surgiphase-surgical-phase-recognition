# Waits for the Cholec80 subset download to finish, then extracts frames, trains and evaluates.
# Progress goes to train.log (shown on the live dashboard). Safe to re-run: finished stages are skipped.
$proj = 'D:\med-ai-portfolio\06-surgical-phase-recognition'
$py = 'D:\med-ai-portfolio\.venv\Scripts\python.exe'
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
