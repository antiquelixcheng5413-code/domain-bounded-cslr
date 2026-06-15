# Windows 10, WSL 2, and Docker Desktop

## Prerequisites

1. 64-bit Windows 10 version 22H2 or a supported Windows 11 version.
2. Hardware virtualization enabled in BIOS/UEFI.
3. Administrator access for initial installation.
4. At least 20 GB free disk space recommended for WSL, Docker images, and Python dependencies.
5. A separate data/download directory on `D:` is recommended for dataset archives.

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

Docker login is not required for this project. Use Docker without signing in unless you need to
pull private images, push images to Docker Hub, or bypass anonymous Docker Hub pull limits.

On this Windows machine, Docker Desktop was moved off `C:`:

- Program: `D:\DockerDesktop`
- Docker data: `D:\DockerDesktopData`
- Docker config and CLI plugins: `D:\DockerDesktopConfig`

See [storage-migration-audit.md](storage-migration-audit.md) before deleting any migration
backup.

## Troubleshooting

- If WSL reports virtualization is disabled, enable Intel VT-x/AMD-V in BIOS/UEFI.
- If `wsl --install` is unavailable, update Windows and follow Microsoft's manual WSL steps.
- If Docker stays on “Starting”, run `wsl --shutdown`, then reopen Docker Desktop.
- Keep code in the Git repository, but keep large datasets and downloaded archives outside Git.
  This project uses `D:\FYP_downloads` as the local download cache and recommends
  `D:\FYP_downloads\data` as the dataset root.
- Mount the external dataset directory through Compose with `CSLR_DATA_ROOT`. Do not copy large
  archives or extracted frames into a Docker image.

Example:

```powershell
$env:CSLR_DATA_ROOT="D:\FYP_downloads\data"
docker compose run --rm dev list-adapters
```
