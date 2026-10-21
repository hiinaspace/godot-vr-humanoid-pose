#!/usr/bin/env bash
# Round 3 RunPod helper (runs on sayu). Credentials are read from ~/code/piano-rotoscope/.env inside each
# command and never printed. Community-cloud 4090, on-demand, absolute terminate-after deadline.
#   r3_pod.sh balance
#   r3_pod.sh up LABEL HOURS            create the pod (waits for ssh), stores the id in .cache/sprint/r3/pod_LABEL
#   r3_pod.sh setup LABEL               copy code (git HEAD) + data, pip install
#   r3_pod.sh start LABEL JOB...        run jobs (names from jobs.json) sequentially, detached, on the pod
#   r3_pod.sh pull LABEL                copy finished runs back (best.pt, last.pt, elapsed_gpu_s, logs)
#   r3_pod.sh status LABEL | down LABEL
set -euo pipefail
REPO=/mnt/s/code/godot-vr-humanoid-pose
ST=$REPO/.cache/sprint/r3
set -a
. "$HOME/code/piano-rotoscope/.env"
set +a
export RUNPOD_API_KEY="$RUNPOD_TOKEN"
rp() { nix shell nixpkgs#runpodctl --command runpodctl "$@"; }
KEY=$HOME/.ssh/id_ed25519
cmd=${1:?}
shift
balance() {
  curl -sS https://api.runpod.io/graphql -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $RUNPOD_TOKEN" -d '{"query":"query { myself { clientBalance } }"}' | jq -r .data.myself.clientBalance
}
podid() { cat "$ST/pod_$1"; }
sshp() {
  local label=$1
  shift
  local info ip port
  info=$(rp ssh info "$(podid "$label")")
  ip=$(jq -r .ip <<<"$info")
  port=$(jq -r .port <<<"$info")
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -i "$KEY" -p "$port" "root@$ip" "$@"
}
sshaddr() {
  local info
  info=$(rp ssh info "$(podid "$1")")
  echo "$(jq -r .ip <<<"$info") $(jq -r .port <<<"$info")"
}
case $cmd in
balance) balance ;;
up)
  label=$1
  hours=$2
  b=$(balance)
  echo "balance before: $b"
  deadline=$(date -u -d "+$(awk "BEGIN{print int($hours*60)}") minutes" +%Y-%m-%dT%H:%M:%SZ)
  body=$(jq -n --arg c "${CLOUD:-COMMUNITY}" --arg g "${GPU:-NVIDIA GeForce RTX 4090}" --arg n "r3-$label" '{name:$n, imageName:"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", gpuTypeIds:[$g], gpuCount:1, cloudType:$c, computeType:"GPU", containerDiskInGb:40, volumeInGb:0, ports:["22/tcp"], supportPublicIp:true}')
  out=$(curl -sS -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_TOKEN" -H 'Content-Type: application/json' -d "$body")
  jq -e .id <<<"$out" >/dev/null || { echo "create failed: $(jq -c 'del(.env)' <<<"$out" | cut -c1-300)"; exit 1; }
  jq -r .id <<<"$out" >"$ST/pod_$label"
  echo "pod $(podid "$label") created; the REST API has no terminate-after, so a sayu timer deletes it at $deadline"
  systemd-run --user --quiet --unit="r3-podkill-$label" --on-active="$(awk "BEGIN{print int($hours*3600)}")s" "$REPO/validation/sprint/r3_pod.sh" down "$label"
  for i in $(seq 60); do
    if info=$(rp ssh info "$(podid "$label")" 2>/dev/null) && [ "$(jq -r '.ip // empty' <<<"$info")" != "" ] && sshp "$label" true 2>/dev/null; then echo "ssh ready"; break; fi
    sleep 15
  done
  ;;
setup)
  label=$1
  read -r ip port < <(sshaddr "$label")
  SSHO="ssh -o StrictHostKeyChecking=accept-new -i $KEY -p $port"
  sshp "$label" 'mkdir -p /workspace/repo/.cache/sprint /workspace/repo/.cache/ml'
  (cd "$REPO" && git archive HEAD validation | $SSHO "root@$ip" 'tar x -C /workspace/repo')
  tar c -C "$REPO/.cache/sprint" data | $SSHO "root@$ip" 'tar x -C /workspace/repo/.cache/sprint'
  sshp "$label" 'pip install -q --ignore-installed blinker polars mlflow scipy 2>&1 | tail -2; python -c "import torch;print(torch.__version__, torch.cuda.is_available())"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2'
  ;;
start)
  label=$1
  shift
  python3 - "$@" >"/tmp/r3_pod_$label.sh" <<'PY'
import json, shlex, sys
jobs = {j["name"]: j for j in json.load(open("/mnt/s/code/godot-vr-humanoid-pose/.cache/sprint/r3/jobs.json"))}
print("#!/usr/bin/env bash\ncd /workspace/repo\nexport PYTHONPATH=.\nmkdir -p logs")
for n in sys.argv[1:]:
    a = " ".join(shlex.quote(x) for x in jobs[n]["train"])
    print(f"python -m validation.sprint.train train --run {n} --time-min 110 {a} >> logs/{n}.log 2>&1; echo EXIT=$? >> logs/{n}.log")
print("echo ALLDONE > logs/ALLDONE")
PY
  read -r ip port < <(sshaddr "$label")
  scp -q -o StrictHostKeyChecking=accept-new -i "$KEY" -P "$port" "/tmp/r3_pod_$label.sh" "root@$ip:/workspace/run.sh"
  sshp "$label" 'chmod +x /workspace/run.sh; nohup setsid /workspace/run.sh > /workspace/run.out 2>&1 < /dev/null & echo started'
  ;;
pull)
  label=$1
  read -r ip port < <(sshaddr "$label")
  SSHO="ssh -o StrictHostKeyChecking=accept-new -i $KEY -p $port"
  mkdir -p "$ST/podlogs"
  $SSHO "root@$ip" 'tar c -C /workspace/repo logs' | tar x -C "$ST/podlogs" --strip-components=1
  for r in $(sshp "$label" 'ls /workspace/repo/.cache/sprint/runs 2>/dev/null'); do
    mkdir -p "$REPO/.cache/sprint/runs/$r"
    # elapsed_gpu_s only exists once a run finished; copy it last so the queue never sees a partial run
    $SSHO "root@$ip" "tar c -C /workspace/repo/.cache/sprint/runs/$r --exclude=elapsed_gpu_s ." | tar x -C "$REPO/.cache/sprint/runs/$r"
    $SSHO "root@$ip" "cat /workspace/repo/.cache/sprint/runs/$r/elapsed_gpu_s" > "$ST/.egs_$r" 2>/dev/null && mv "$ST/.egs_$r" "$REPO/.cache/sprint/runs/$r/elapsed_gpu_s" || rm -f "$ST/.egs_$r"
  done
  ;;
chain)
  # replace the pod's pending job list: kill the runner bash (not the running python), then run JOBs after it ends
  label=$1
  shift
  python3 - "$@" >"/tmp/r3_pod2_$label.sh" <<'PY'
import json, shlex, sys
jobs = {j["name"]: j for j in json.load(open("/mnt/s/code/godot-vr-humanoid-pose/.cache/sprint/r3/jobs.json"))}
print("#!/usr/bin/env bash\ncd /workspace/repo\nexport PYTHONPATH=.\nmkdir -p logs")
print('PY=$(ps -eo pid,args | awk \'/validation.sprint.train train/ && !/awk/ {print $1; exit}\')')
print('while [ -n "$PY" ] && kill -0 $PY 2>/dev/null; do sleep 20; done')
for n in sys.argv[1:]:
    a = " ".join(shlex.quote(x) for x in jobs[n]["train"])
    print(f"python -m validation.sprint.train train --run {n} --time-min 110 {a} >> logs/{n}.log 2>&1; echo EXIT=$? >> logs/{n}.log")
print("echo ALLDONE > logs/ALLDONE")
PY
  read -r ip port < <(sshaddr "$label")
  scp -q -o StrictHostKeyChecking=accept-new -i "$KEY" -P "$port" "/tmp/r3_pod2_$label.sh" "root@$ip:/workspace/run2.sh"
  sshp "$label" 'RP=$(ps -eo pid,args | awk "/workspace\/run.sh/ && !/awk/ {print \$1; exit}"); echo old runner pid $RP; [ -n "$RP" ] && kill $RP; chmod +x /workspace/run2.sh; nohup setsid /workspace/run2.sh > /workspace/run2.out 2>&1 < /dev/null & echo chained'
  ;;
status)
  label=$1
  sshp "$label" 'tail -n 2 /workspace/repo/logs/*.log 2>/dev/null | cut -c1-160; ls /workspace/repo/logs'
  ;;
down)
  label=$1
  rp pod delete "$(podid "$label")" || true
  sleep 5
  rp pod list -o json
  echo "balance after: $(balance)"
  ;;
esac
