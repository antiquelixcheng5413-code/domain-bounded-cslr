# Storage migration audit

Audit date: 2026-06-16.

This file records the Windows storage decisions that should be checked before future cleanup.
The goal is to keep large Docker, dataset, and tool artifacts off the system `C:` drive.

## Current layout

| Purpose | Current path | Approximate size at migration | Notes |
|---|---|---:|---|
| Project Git repository | `E:\college\FYP` | small | Source, configs, tests, and docs only |
| Docker Desktop program | `D:\DockerDesktop` | 3.38 GB | Installed with Docker installer `--installation-dir` |
| Docker images, containers, build cache | `D:\DockerDesktopData` | 15.53 GB | Installed/restored with `--wsl-default-data-root` |
| Docker user config and CLI plugins | `D:\DockerDesktopConfig` | 0.67 GB | C user paths are junctions into this folder |
| Docker migration backup | `D:\DockerDesktopBackup` | 17.16 GB | Keep until Docker has been rechecked after restarts |
| Dataset and installer downloads | `D:\FYP_downloads` | 0.89 GB | Includes NationalCSL-DP audit files and Docker installer |
| Portable helper tools | `D:\FYP_tools` | 0.04 GB | Includes portable GitHub CLI |

## C-drive junctions

These C-drive paths are intentionally left as junctions so Docker keeps working while the
files live on `D:`:

| C path | Target |
|---|---|
| `C:\Users\CN\.docker` | `D:\DockerDesktopConfig\dot-docker` |
| `C:\Users\CN\AppData\Local\Docker` | `D:\DockerDesktopConfig\Local-Docker` |
| `C:\Users\CN\AppData\Roaming\Docker` | `D:\DockerDesktopConfig\Roaming-Docker` |
| `C:\Users\CN\AppData\Roaming\Docker Desktop` | `D:\DockerDesktopConfig\Roaming-DockerDesktop` |
| `E:\college\FYP\.downloads` | `D:\FYP_downloads` |
| `E:\college\FYP\.tools` | `D:\FYP_tools` |

## Verified after migration

- `docker.exe` resolves to `D:\DockerDesktop\resources\bin\docker.exe`.
- Docker Engine starts successfully.
- Existing project images are still present: `fyp-dev`, `fyp-test`, and `fyp-app`.
- The old C-drive install path `C:\Users\CN\AppData\Local\Programs\DockerDesktop` does not
  exist.
- The old C-drive Docker data disk
  `C:\Users\CN\AppData\Local\Docker\wsl\disk\docker_data.vhdx` does not exist.
- `C:\Program Files\WSL` remains on C because it is a Windows system component, not a project
  cache. Its size was about 0.807 GB during this audit.

## Backup cleanup rule

Do not delete `D:\DockerDesktopBackup` immediately. It contains the pre-migration Docker WSL
data and a fresh blank data directory created during reinstall.

It is safe to remove only after all checks below are true:

1. Windows has been restarted at least once.
2. Docker Desktop opens from `D:\DockerDesktop`.
3. `docker version` succeeds.
4. `docker images` still shows the expected project images or the project has been rebuilt.
5. `docker compose --profile test run --rm test` passes from `E:\college\FYP`.
6. No required file is missing from `D:\DockerDesktopData`.

After those checks, delete only the backup folder:

```powershell
Remove-Item -LiteralPath "D:\DockerDesktopBackup" -Recurse -Force
```

Do not delete `D:\DockerDesktop`, `D:\DockerDesktopData`, `D:\DockerDesktopConfig`,
`D:\FYP_downloads`, or `D:\FYP_tools` while the project is active.
