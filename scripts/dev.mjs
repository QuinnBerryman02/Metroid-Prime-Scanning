#!/usr/bin/env node
/**
 * One command to run the whole stack:
 *
 *   npm run dev
 *
 * Starts the inference service, the gateway, and the Vite client together,
 * prefixing each line of output with its source. Ctrl+C stops everything.
 * Cross-platform (Windows / macOS / Linux), no extra dependencies.
 *
 * The gateway and client are launched by running their tools' JS entry points
 * with node directly rather than via `npm run --workspace`. Going through npm
 * on Windows wraps the command in a cmd.exe batch job, which makes Ctrl+C pop
 * up a "Terminate batch job (Y/N)?" prompt. Spawning node directly avoids it.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const isWin = process.platform === "win32";
const require = createRequire(import.meta.url);

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const COLOURS = {
  inference: "\x1b[35m", // magenta
  gateway: "\x1b[36m", // cyan
  client: "\x1b[32m", // green
};

/** Absolute path to a file inside an installed package (e.g. tsx's CLI). */
function packageFile(pkg, ...rel) {
  return join(dirname(require.resolve(`${pkg}/package.json`)), ...rel);
}

/** Prefer the project venv, whichever layout it uses. */
function findPython() {
  const rel = isWin
    ? [".venv/Scripts/python.exe", ".venv/Scripts/python"]
    : [".venv/bin/python", ".venv/bin/python3"];
  for (const r of rel) {
    const p = join(root, "services", "inference", r);
    if (existsSync(p)) return p;
  }
  return null;
}

const tasks = [];

const python = findPython();
if (python) {
  tasks.push({
    name: "inference",
    cmd: python,
    args: ["-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
    cwd: join(root, "services", "inference"),
  });
} else {
  const bin = isWin ? ".venv\\Scripts\\python" : ".venv/bin/python";
  console.warn(
    `${DIM}[inference] no venv at services/inference/.venv — skipping it.${RESET}\n` +
      `${DIM}           set it up with:\n` +
      `             cd services/inference\n` +
      `             python -m venv .venv\n` +
      `             ${bin} -m pip install -r requirements.txt${RESET}`,
  );
}

try {
  tasks.push({
    name: "gateway",
    cmd: process.execPath,
    args: [packageFile("tsx", "dist", "cli.mjs"), "watch", "src/index.ts"],
    cwd: join(root, "apps", "server"),
  });
  tasks.push({
    name: "client",
    cmd: process.execPath,
    args: [packageFile("vite", "bin", "vite.js")],
    cwd: join(root, "apps", "client"),
  });
} catch (err) {
  console.error(
    `\nCould not find tsx/vite — run "npm install" first.\n${err.message}\n`,
  );
  process.exit(1);
}

const children = [];
let shuttingDown = false;

function pipe(stream, name) {
  const colour = COLOURS[name] ?? "";
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      process.stdout.write(`${colour}[${name}]${RESET} ${line}\n`);
    }
  });
  stream.on("end", () => {
    if (buffer) process.stdout.write(`${colour}[${name}]${RESET} ${buffer}\n`);
  });
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write(`\n${DIM}Shutting down…${RESET}\n`);
  for (const child of children) {
    if (child.exitCode !== null || child.signalCode !== null) continue;
    try {
      if (isWin) {
        spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
          stdio: "ignore",
        });
      } else {
        child.kill("SIGTERM");
      }
    } catch {
      /* ignore */
    }
  }
  // Give children a moment to die, then exit.
  setTimeout(() => process.exit(code), 500);
}

console.log("Starting full stack — inference + gateway + client (Ctrl+C to stop)\n");

for (const task of tasks) {
  const child = spawn(task.cmd, task.args, {
    cwd: task.cwd,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  children.push(child);
  pipe(child.stdout, task.name);
  pipe(child.stderr, task.name);
  child.on("error", (err) => {
    console.error(`${COLOURS[task.name] ?? ""}[${task.name}]${RESET} ${err.message}`);
    shutdown(1);
  });
  child.on("exit", (code) => {
    if (shuttingDown) return;
    process.stdout.write(
      `${COLOURS[task.name] ?? ""}[${task.name}]${RESET} exited (code ${code})\n`,
    );
    shutdown(code ?? 1);
  });
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
