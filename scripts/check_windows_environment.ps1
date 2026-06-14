$ErrorActionPreference = "Continue"

Write-Host "Domain-Bounded CSLR Windows environment check" -ForegroundColor Cyan
Write-Host ""

$windows = Get-ComputerInfo -Property WindowsProductName, WindowsVersion, OsArchitecture `
  -ErrorAction SilentlyContinue
if ($windows) {
  $windows | Format-List
} else {
  Write-Warning "Windows information could not be read with the current permissions."
}

Write-Host "WSL status" -ForegroundColor Cyan
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
  wsl.exe --status
} else {
  Write-Warning "wsl.exe is unavailable. Run 'wsl --install' in Administrator PowerShell."
}

Write-Host ""
Write-Host "Docker status" -ForegroundColor Cyan
if (Get-Command docker.exe -ErrorAction SilentlyContinue) {
  docker version
  docker compose version
} else {
  Write-Warning "Docker Desktop is not installed or is not on PATH."
}

Write-Host ""
Write-Host "Manual check required:" -ForegroundColor Yellow
Write-Host "Open Task Manager > Performance > CPU and confirm Virtualization is Enabled."
Write-Host "Installation guide: docs/setup/windows-docker.md"
