---
name: gpu-remote-exec
description: Run commands or CUDA-backed Python scripts on the remote GPU box via ssh root@gpu-box when execution needs a GPU/CUDA backend or the user asks to run on the GPU box.
---

# GPU Remote Exec

Use this skill to execute commands on the CUDA-capable GPU box at `root@gpu-box`.

## Core rule

* **Edit code only locally. Never edit files on the remote GPU box.**
* After **any** local code change (even a single line), **re-sync the local diff to the remote** and then run remotely.

## Workflow

0. Decide if the task requires GPU/CUDA or the user explicitly asks for the GPU box.

1. Check which remote gpu answers and use it until it goes down, test with:
```bash
ssh root@gpu-box "echo ok"
ssh root@gpu-box-1 "echo ok"
ssh root@gpu-box-2 "echo ok"
ssh root@gpu-box-3 "echo ok"
ssh root@gpu-box-4 "echo ok"
```

2. Ensure the remote repo exists and is on the correct branch (run once per session, or if branch changes):

```bash
ssh root@gpu-box 'set -e; cd /proj/patatnik; git fetch --all --prune'
ssh root@gpu-box 'set -e; cd /proj/patatnik; git status --porcelain; git branch --show-current'
```

3. **After every local edit**, force the remote into a clean state and apply your **local** changes via a binary diff.

**A) Include new files in the diff (recommended):**

```bash
git add -N .
git diff --binary | ssh root@gpu-box 'set -e; cd /proj/patatnik; git reset --hard; git clean -fd; git apply --index'
```

**B) If you only changed already-tracked files:**

```bash
git diff --binary | ssh root@gpu-box 'set -e; cd /proj/patatnik; git reset --hard; git clean -fd; git apply --index'
```

4. Run the desired command or script remotely:

```bash
ssh root@gpu-box 'set -e; cd /proj/patatnik; <RUN_COMMAND_HERE>'
```
> Note: the base image provides Python at `/opt/venv/bin/python`. Use it explicitly because the venv over ssh is not auto-activated.

5. If results require code changes:

   * Make the change locally.
   * Repeat **Step 3** (sync diff).
   * Repeat **Step 4** (run remotely).

6. Clean up the remote box (optional, but good hygiene at the end of a session):

```bash
ssh root@gpu-box 'set -e; cd /proj/patatnik; git reset --hard; git clean -fd'
```

## Notes

* The remote is treated as **ephemeral**: it is always reset/cleaned before applying the local patch.
* If `git apply` fails, it usually means the remote is not on a compatible base revision or branch. Fix by aligning the remote branch/checkout and then re-run the sync command in Step 3.
* Always use the diff-send method above to run local work-in-progress remotely; do not try to edit remotely.
* The base image is intentionally minimal. If the project needs extra Python packages, install them explicitly on the pod or extend the Docker image.
