---
name: deploy-gpu
description: Create/start/stop/delete a RunPod GPU Pod using ONLY GPU type `XXX` + the existing template named `PATATNIK-container`. Includes Spot (Secure Cloud) via interruptible Pods.
---

# RunPod PATATNIK Pod

Use this skill to manage Secure Cloud RunPod pods from template `PATATNIK-container` with GPU `XXX`.

**Read your `remote-gpu` skill after this one to know how to use the newly created pod.**

## Core rules

* Secure Cloud only.
* Set only GPU type + template id (no extra tuning unless asked).
* Use the existing template named `PATATNIK-container`.
* Prefer CLI for on-demand pods.
* If CLI fails with `required flag(s) "imageName" not set` while using `--templateId`, use REST create.
* Spot means `interruptible: true` via REST.

## Workflow

1. Resolve the template id for `PATATNIK-container` and export it:

```bash
runpodctl get template | rg 'PATATNIK-container'
PATATNIK_TEMPLATE_ID="<paste-template-id-here>"
```

2. Create on-demand pod (CLI):

```bash
runpodctl create pod --secureCloud --gpuType "XXX" --templateId "$PATATNIK_TEMPLATE_ID"
```

3. If CLI needs `--imageName`, create via REST:

```bash
curl --silent --show-error \
  --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{"cloudType":"SECURE","computeType":"GPU","gpuCount":1,"gpuTypeIds":["XXX"],"templateId":"'"$PATATNIK_TEMPLATE_ID"'"}'
```

For Spot, add `"interruptible": true` to the same payload.

4. Wait for the pod to be fully usable - run this script:
`CICD/tools/wait-new-gpu.sh`

This script sleeps until it reads a signal that the gpu is ready. Only when the script successfully finishes should you use the pod with the workflow in your gpu skill.

5. Manage pod lifecycle:

```bash
runpodctl get pod
runpodctl stop pod "$RUNPOD_POD_ID"
runpodctl remove pod "$RUNPOD_POD_ID"
```

# What to do when the Pod starts

After the pod is created and it has started the container, it will be available to connect to at some `root@gpu-box-N` id. Run `tailscale status | rg gpu` to see its id.

You should run commands on the remote gpu only like mentioned in your `remote-gpu` skill.

# What to do when the Pod stops / you are terminating it

Pods are **ephemeral** resources. Copy out any logs, checkpoints, datasets, or outputs you care about from `/proj` before you stop or remove the pod.

## Notes

* Spot pods are interruptible and can terminate. Use spot only if mentioned by the user.
* For quick testing use gpu `NVIDIA A40`, for heavier runs use something larger only when needed.
