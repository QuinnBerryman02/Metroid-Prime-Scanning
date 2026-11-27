#!/usr/bin/env node
/**
 * Kill anything left over from the metroid-scan stack:
 *
 *   npm run stop
 *
 * `npm run dev` stops its children on Ctrl+C, but a crash, a closed terminal or
 * a re-parented uvicorn worker can leave processes behind holding ports 3000,
 * 5173 and 8000. This finds them three ways and kills each process tree:
 *
 *   1. whatever is LISTENING on the stack's ports (3000 gateway, 5173 client,
 *      8000 inference), whatever started it;
 *   2. processes whose command line is one of ours (uvicorn app.main:app,
 *      tsx watch src/index.ts, vite, scripts/dev.mjs);
 *   3. orphaned multiprocessing workers (uvicorn --reload's children) whose
 *      parent has already exited.
 *
 * It does not touch Jupyter kernels or other editors' Python processes.
 */
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PORTS = [3000, 5173, 8000];
const isWin = process.platform === "win32";

function killWindows() {
  const ps = `
$ports = @(${PORTS.join(",")})
$found = @{}
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $ports -contains $_.LocalPort } |
  ForEach-Object { $found[[int]$_.OwningProcess] = "port $($_.LocalPort)" }
$procs = @(Get-CimInstance Win32_Process)
$alive = @{}
foreach ($p in $procs) { $alive[[int]$p.ProcessId] = $true }
foreach ($p in $procs) {
  $cl = [string]$p.CommandLine
  if ($cl -match 'app\\.main:app|watch src/index\\.ts|vite\\.js|scripts.dev\\.mjs') {
    $found[[int]$p.ProcessId] = "ours: " + $p.Name
  }
  if ($cl -match 'multiprocessing\\.spawn' -and -not $alive.ContainsKey([int]$p.ParentProcessId)) {
    $found[[int]$p.ProcessId] = "orphaned worker"
  }
}
$found.Remove($PID)
$found.GetEnumerator() | ForEach-Object { "$($_.Key)|$($_.Value)" }
`;
  // A script file, not stdin: `powershell -Command -` silently prints nothing.
  const dir = mkdtempSync(join(tmpdir(), "stop-"));
  const file = join(dir, "find.ps1");
  writeFileSync(file, ps);
  const out = spawnSync(
    "powershell",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", file],
    { encoding: "utf8" },
  ).stdout ?? "";
  rmSync(dir, { recursive: true, force: true });
  const targets = out
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => /^\d+\|/.test(l))
    .map((l) => {
      const [pid, why] = l.split("|");
      return { pid: Number(pid), why };
    });
  return targets;
}

function killUnix() {
  const targets = new Map();
  for (const port of PORTS) {
    try {
      const pids = execFileSync("lsof", ["-ti", `tcp:${port}`], { encoding: "utf8" });
      for (const pid of pids.split(/\s+/).filter(Boolean)) targets.set(Number(pid), `port ${port}`);
    } catch {
      /* nothing on that port */
    }
  }
  try {
    const ps = execFileSync("ps", ["-eo", "pid=,args="], { encoding: "utf8" });
    for (const line of ps.split("\n")) {
      const m = line.trim().match(/^(\d+)\s+(.*)$/);
      if (!m) continue;
      if (/app\.main:app|watch src\/index\.ts|vite\.js|scripts\/dev\.mjs/.test(m[2])) {
        targets.set(Number(m[1]), "ours");
      }
    }
  } catch {
    /* ps unavailable */
  }
  targets.delete(process.pid);
  return [...targets].map(([pid, why]) => ({ pid, why }));
}

const targets = isWin ? killWindows() : killUnix();
if (targets.length === 0) {
  console.log("Nothing from the stack is running.");
  process.exit(0);
}

for (const { pid, why } of targets) {
  console.log(`stopping ${pid} (${why})`);
  if (isWin) {
    // /T kills the whole tree, so a worker goes with its parent.
    spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
}

// Confirm the ports are free.
const still = isWin
  ? killWindows().length
  : killUnix().length;
console.log(still === 0 ? "Done. Ports 3000, 5173, 8000 are free." : `${still} process(es) still running - run npm run stop again.`);
