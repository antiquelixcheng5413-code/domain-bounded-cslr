# Git and GitHub workflow

## Repository setup

The local repository uses `main`. Create a private empty GitHub repository, then connect it:

```powershell
git remote add origin https://github.com/OWNER/domain-bounded-cslr.git
git push -u origin main
```

Do not add a remote until the repository URL and account owner are confirmed.

## Branch protection

In GitHub settings for `main`:

- Require a pull request before merging.
- Require the `quality` and `docker` status checks.
- Block force pushes and branch deletion.
- Require conversation resolution.

## Releases

Create a signed or annotated tag only after the matching experiment record is complete:

```powershell
git tag -a v0.1.0 -m "Week 12 controlled prototype"
git push origin v0.1.0
```

The release workflow publishes the private container image. Attach the ONNX model,
`*.labels.json`, checksums, model card, and metrics to the private GitHub Release.
