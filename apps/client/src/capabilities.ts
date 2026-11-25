/**
 * WebXR capability / depth-sensing diagnostic.
 *
 * Manual test page (reachable at /capabilities.html). Run it on the phone over
 * HTTPS and it will:
 *   1. report secure-context + WebXR basics,
 *   2. request an `immersive-ar` session with every feature we care about as
 *      optional, then report which ones the runtime actually enabled
 *      (`session.enabledFeatures`),
 *   3. enter a minimal WebGL AR loop that live-reports hit-test results and
 *      depth-sensing data (dimensions, usage, and a depth preview).
 *
 * No Babylon dependency on purpose - this must run even if the app code breaks.
 */

const FEATURES = [
  "hit-test",
  "anchors",
  "plane-detection",
  "depth-sensing",
  "camera-access",
  "dom-overlay",
  "light-estimation",
  "hand-tracking",
  "layers",
  "local-floor",
  "bounded-floor",
  "unbounded",
] as const;

type FeatureName = (typeof FEATURES)[number];

// --- tiny DOM helpers -------------------------------------------------------

const runBtn = document.getElementById("runBtn") as HTMLButtonElement;
const copyBtn = document.getElementById("copyBtn") as HTMLButtonElement;
const exitBtn = document.getElementById("exitBtn") as HTMLButtonElement;
const envEl = document.getElementById("env")!;
const featuresEl = document.getElementById("features")!;
const sessionEl = document.getElementById("session")!;
const logEl = document.getElementById("log")!;
const hudEl = document.getElementById("hud")!;
const reticleEl = document.getElementById("reticle")!;
const depthCanvas = document.getElementById("depthPreview") as HTMLCanvasElement;

function row(name: string, value: string, cls = ""): string {
  return `<div class="row"><span class="name">${name}</span><span class="${cls}">${value}</span></div>`;
}
function setEnv(html: string) {
  envEl.innerHTML = html;
}
function setSession(html: string) {
  sessionEl.innerHTML = html;
}
function log(msg: string) {
  const t = new Date().toISOString().slice(11, 19);
  logEl.textContent += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function copyReport() {
  const report = [
    envEl.textContent,
    featuresEl.textContent,
    sessionEl.textContent,
    logEl.textContent,
  ].join("\n");
  navigator.clipboard?.writeText(report).then(
    () => (copyBtn.textContent = "Copied!"),
    () => (copyBtn.textContent = "Copy failed"),
  );
  setTimeout(() => (copyBtn.textContent = "Copy report"), 1500);
}

// --- capability probing -----------------------------------------------------

async function probeEnvironment(): Promise<boolean> {
  const xrSystem = (navigator as Navigator & { xr?: XRSystem }).xr;
  const secure = window.isSecureContext;

  let arSupported = false;
  if (xrSystem) {
    try {
      arSupported = await xrSystem.isSessionSupported("immersive-ar");
    } catch (e) {
      log(`isSessionSupported error: ${String(e)}`);
    }
  }

  setEnv(
    row("secure context (https)", secure ? "yes" : "NO", secure ? "yes" : "no") +
      row("navigator.xr", xrSystem ? "present" : "missing", xrSystem ? "yes" : "no") +
      row(
        "immersive-ar",
        xrSystem ? (arSupported ? "supported" : "not supported") : "n/a",
        arSupported ? "yes" : "no",
      ) +
      row("user agent", navigator.userAgent.slice(0, 60)),
  );

  log(`secure=${secure} xr=${!!xrSystem} immersive-ar=${arSupported}`);
  return secure && arSupported;
}

// --- AR session test --------------------------------------------------------

let session: XRSession | null = null;

async function startSessionTest() {
  const xrSystem = (navigator as Navigator & { xr: XRSystem }).xr;
  runBtn.disabled = true;
  featuresEl.innerHTML = "";

  try {
    session = await xrSystem.requestSession("immersive-ar", {
      requiredFeatures: ["hit-test"],
      optionalFeatures: [...FEATURES],
      domOverlay: { root: document.getElementById("overlay")! as HTMLElement },
    });
  } catch (e) {
    log(`requestSession failed: ${String(e)}`);
    featuresEl.innerHTML = row("session", `FAILED: ${String(e)}`, "no");
    runBtn.disabled = false;
    return;
  }

  log(`session started (blend=${session.environmentBlendMode})`);
  document.body.classList.add("in-ar");

  // `enabledFeatures` is the cleanest way to know what was actually granted.
  const enabled = (session as XRSession & { enabledFeatures?: string[] })
    .enabledFeatures;
  const enabledSet = new Set(enabled ?? []);
  featuresEl.innerHTML = FEATURES.map((f) => {
    if (f === "hit-test") return row(f, "required (granted)", "yes");
    if (!enabled) return row(f, "unknown (runtime has no enabledFeatures)", "warn");
    return enabledSet.has(f) ? row(f, "enabled", "yes") : row(f, "unavailable", "no");
  }).join("");

  reportSessionDetails();

  session.addEventListener("end", () => {
    document.body.classList.remove("in-ar");
    runBtn.disabled = false;
    session = null;
    log("session ended");
  });

  await runArLoop(session);
}

function reportSessionDetails() {
  if (!session) return;
  const s = session as XRSession & {
    depthUsage?: string;
    depthDataFormat?: string;
    preferredReflectionFormat?: string;
  };
  setSession(
    row("interaction mode", s.interactionMode ?? "n/a") +
      row("environment blend", s.environmentBlendMode ?? "n/a") +
      row("depth usage", s.depthUsage ?? "n/a", s.depthUsage ? "yes" : "no") +
      row(
        "depth data format",
        s.depthDataFormat ?? "n/a",
        s.depthDataFormat ? "yes" : "no",
      ) +
      row("reflection format", s.preferredReflectionFormat ?? "n/a"),
  );
  log(
    `interaction=${s.interactionMode} blend=${s.environmentBlendMode} ` +
      `depthUsage=${s.depthUsage ?? "-"} depthFormat=${s.depthDataFormat ?? "-"}`,
  );
}

async function runArLoop(s: XRSession) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", {
    xrCompatible: true,
    alpha: true,
    antialias: true,
  });
  if (!gl) {
    log("webgl2 unavailable");
    return;
  }
  await gl.makeXRCompatible();

  const layer = new XRWebGLLayer(s, gl);
  s.updateRenderState({ baseLayer: layer, depthNear: 0.1, depthFar: 100 });

  let refSpace: XRReferenceSpace;
  try {
    refSpace = (await s.requestReferenceSpace("local")) as XRReferenceSpace;
  } catch (e) {
    log(`requestReferenceSpace failed: ${String(e)}`);
    return;
  }

  let hitTestSource: XRHitTestSource | null = null;
  const requestHitTest = s.requestHitTestSource?.bind(s);
  if (requestHitTest) {
    try {
      hitTestSource = (await requestHitTest({ space: refSpace })) ?? null;
      log("hit-test source acquired");
    } catch (e) {
      log(`requestHitTestSource failed: ${String(e)}`);
    }
  } else {
    log("requestHitTestSource not available on this session");
  }

  const depthCtx = depthCanvas.getContext("2d")!;
  let lastDepthDraw = 0;
  let depthReported = false;
  let frames = 0;

  const onFrame: XRFrameRequestCallback = (time, frame) => {
    s.requestAnimationFrame(onFrame);
    frames++;

    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const pose = frame.getViewerPose(refSpace);
    if (!pose) return;

    // --- hit test ---
    let hits = 0;
    if (hitTestSource) {
      const results = frame.getHitTestResults(hitTestSource);
      hits = results.length;
      reticleEl.classList.toggle("hit", hits > 0);
    }

    // --- depth ---
    let depthText = "depth: none on this device";
    for (const view of pose.views) {
      const depth = frame.getDepthInformation?.(view) as XRDepthInformation | null;
      if (!depth) continue;
      const cpu = depth as unknown as { getDepthInMeters?: (x: number, y: number) => number };
      let centre = "";
      if (typeof cpu.getDepthInMeters === "function") {
        try {
          const m = cpu.getDepthInMeters(depth.width / 2, depth.height / 2);
          centre = ` centre=${m.toFixed(2)}m`;
        } catch {
          /* out of range */
        }
      }
      depthText = `depth: ${depth.width}x${depth.height} ${(session as XRSession & { depthUsage?: string }).depthUsage ?? ""}${centre}`;
      if (!depthReported) {
        depthReported = true;
        log(`depth-sensing ACTIVE: ${depth.width}x${depth.height}${centre}`);
      }
      if (time - lastDepthDraw > 250 && cpu.getDepthInMeters) {
        lastDepthDraw = time;
        drawDepthPreview(cpu.getDepthInMeters.bind(depth));
      }
      void depthCtx;
    }

    hudEl.textContent =
      `frame ${frames}\n` +
      `views ${pose.views.length}\n` +
      `hit-test hits ${hits}${hits ? " (reticle locked)" : ""}\n` +
      depthText;
  };

  s.requestAnimationFrame(onFrame);
  log("render loop running");
}

function drawDepthPreview(sample: (x: number, y: number) => number) {
  const w = depthCanvas.width;
  const h = depthCanvas.height;
  const ctx = depthCanvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(w, h);
  // The depth map size may differ from the canvas; sample over its own extent.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = NaN;
      try {
        m = sample(x, y);
      } catch {
        /* ignore */
      }
      const v = Number.isFinite(m) ? Math.max(0, Math.min(1, 1 - m / 5)) : 0;
      const i = (y * w + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// --- wiring -----------------------------------------------------------------

async function init() {
  copyBtn.addEventListener("click", copyReport);
  runBtn.addEventListener("click", () => void startSessionTest());
  exitBtn.addEventListener("click", () => void session?.end());

  const canRun = await probeEnvironment();
  runBtn.disabled = !canRun;
  runBtn.textContent = canRun
    ? "Run AR capability test"
    : "AR not supported on this browser/device";
  if (!canRun) {
    log("Cannot start immersive-ar session here.");
  }
}

void init();
