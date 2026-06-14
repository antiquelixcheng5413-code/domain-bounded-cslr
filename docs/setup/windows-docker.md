# Windows 10, WSL 2, and Docker Desktop

## Prerequisites

1. 64-bit Windows 10 version 22H2 or a supported Windows 11 version.
2. Hardware virtualization enabled in BIOS/UEFI.
3. Administrator access for initial installation.
4. At least 20 GB free disk space recommended for WSL, Docker images, and Python dependencies.

## Installation

Run in Administrator PowerShell:

```powershell
wsl --install
```

Restart Windows. Then:

```powershell
wsl --status
```

Install Docker Desktop and select the WSL 2 backend. Verify:

```powershell
docker version
docker run --rm hello-world
docker compose version
```

## Troubleshooting

- If WSL reports virtualization is disabled, enable Intel VT-x/AMD-V in BIOS/UEFI.
- If `wsl --install` is unavailable, update Windows and follow Microsoft's manual WSL steps.
- If Docker stays on “Starting”, run `wsl --shutdown`, then reopen Docker Desktop.
- Keep datasets on the Windows project drive and mount them through Compose. Do not copy them
  into an image.
