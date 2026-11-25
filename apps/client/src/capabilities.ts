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
 * The report is POSTed to `/__diag/report`, which the Vite dev server (see
 * vite.config.ts) writes to `apps/client/reports/`. No manual copying needed.
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

const REPORT_URL = "/__diag/report";

interface DepthInfo {
  active: boolean;
  width?: number;
  height?: number;
  usage?: string;
  format?: string;
  centreMeters?: number;
}

interface ReportData {
  timestamp: string;
  userAgent: string;
  secureContext: boolean;
  hasXR: boolean;
  immersiveArSupported: boolean;
  enabledFeatures: string[] | null;
  session: Record<string, string | null>;
  depth: DepthInfo;
  log: string[];
}

// --- tiny DOM helpers -------------------------------------------------------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const runBtn = $<HTMLButtonElement>("runBtn");
const arSendBtn = $<HTMLButtonElement>("arSendBtn");
const arExitBtn = $<HTMLButtonElement>("arExitBtn");
const envEl = $("env");
const featuresEl = $("features");
const sessionEl = $("session");
const logEl = $("log");
const arFeaturesEl = $("arFeatures");
const arLiveEl = $("arLive");
const reticleEl = $("reticle");
const reticleLabel = $("reticleLabel");
const depthWrap = $("depthWrap");
const depthCanvas = $<HTMLCanvasElement>("depthPreview");

const report: ReportData = {
  timestamp: new Date().toISOString(),
  userAgent: navigator.userAgent,
  secureContext: false,
  hasXR: false,
  immersiveArSupported: false,
  enabledFeatures: null,
  session: {},
  depth: { active: false },
  log: [],
};

function row(name: string, value: string, cls = ""): string {
  return `<div class="row"><span class="name">${name}</span><span class="${cls}">${value}</span></div>`;
}
function log(msg: string) {
  const t = new Date().toISOString().slice(11, 19);
  report.log.push(`[${t}] ${msg}`);
  logEl.textContent = report.log.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}
function setArLive(msg: string) {
  arLiveEl.textContent = msg;
}
function renderFeatureRows(html: string) {
  arFeaturesEl.innerHTML = html;
}

// --- capability probing -----------------------------------------------------

async function probeEnvironment(): Promise<boolean> {
  const xrSystem = (navigator as Navigator & { xr?: XRSystem }).xr;
  const secure = window.isSecureContext;
  report.secureContext = secure;
  report.hasXR = !!xrSystem;

  let arSupported = false;
  if (xrSystem) {
    try {
      arSupported = await xrSystem.isSessionSupported("immersive-ar");
    } catch (e) {
      log(`isSessionSupported error: ${String(e)}`);
    }
  }
  report.immersiveArSupported = arSupported;

  envEl.innerHTML =
    row("secure context (https)", secure ? "yes" : "NO", secure ? "yes" : "no") +
    row("navigator.xr", xrSystem ? "present" : "missing", xrSystem ? "yes" : "no") +
    row(
      "immersive-ar",
      xrSystem ? (arSupported ? "supported" : "not supported") : "n/a",
      arSupported ? "yes" : "no",
    ) +
    row("user agent", navigator.userAgent.slice(0, 60));

  log(`secure=${secure} xr=${!!xrSystem} immersive-ar=${arSupported}`);
  return secure && arSupported;
}

// --- AR session test --------------------------------------------------------

let session: XRSession | null = null;

async function startSessionTest() {
  const xrSystem = (navigator as Navigator & { xr: XRSystem }).xr;
  runBtn.disabled = true;
  featuresEl.innerHTML = "";
  renderFeatureRows('<div class="feat">requesting&hellip;</div>');
  report.timestamp = new Date().toISOString();

  try {
    session = await xrSystem.requestSession("immersive-ar", {
      requiredFeatures: ["hit-test"],
      optionalFeatures: [...FEATURES],
      domOverlay: { root: $("overlay") },
    });
  } catch (e) {
    log(`requestSession failed: ${String(e)}`);
    featuresEl.innerHTML = row("session", `FAILED: ${String(e)}`, "no");
    renderFeatureRows(`<div class="feat no">session failed</div>`);
    runBtn.disabled = false;
    return;
  }

  log(`session started (blend=${session.environmentBlendMode})`);
  document.body.classList.add("in-ar");

  const enabled = (session as XRSession & { enabledFeatures?: string[] })
    .enabledFeatures;
  report.enabledFeatures = enabled ?? null;
  const enabledSet = new Set(enabled ?? []);

  const featureRows = FEATURES.map((f) => {
    if (f === "hit-test") {
      return row(f, "required (granted)", "yes");
    }
    if (!enabled) {
      return row(f, "unknown (no enabledFeatures)", "warn");
    }
    return enabledSet.has(f) ? row(f, "enabled", "yes") : row(f, "unavailable", "no");
  }).join("");
  featuresEl.innerHTML = featureRows;

  renderFeatureRows(
    FEATURES.map((f) => {
      if (f === "hit-test") return `<div class="feat"><span class="name">${f}</span><span class="yes">yes</span></div>`;
      if (!enabled) return `<div class="feat"><span class="name">${f}</span><span class="warn">?</span></div>`;
      return enabledSet.has(f)
        ? `<div class="feat"><span class="name">${f}</span><span class="yes">yes</span></div>`
        : `<div class="feat"><span class="name">${f}</span><span class="no">no</span></div>`;
    }).join(""),
  );

  const s = session as XRSession & { depthUsage?: string; depthDataFormat?: string };
  report.session = {
    interactionMode: s.interactionMode ?? null,
    environmentBlendMode: s.environmentBlendMode ?? null,
    depthUsage: s.depthUsage ?? null,
    depthDataFormat: s.depthDataFormat ?? null,
  };
  sessionEl.innerHTML =
    row("interaction mode", s.interactionMode ?? "n/a") +
    row("environment blend", s.environmentBlendMode ?? "n/a") +
    row("depth usage", s.depthUsage ?? "n/a", s.depthUsage ? "yes" : "no") +
    row("depth data format", s.depthDataFormat ?? "n/a", s.depthDataFormat ? "yes" : "no");
  log(
    `interaction=${s.interactionMode} blend=${s.environmentBlendMode} ` +
      `depthUsage=${s.depthUsage ?? "-"} depthFormat=${s.depthDataFormat ?? "-"}`,
  );

  session.addEventListener("end", () => {
    document.body.classList.remove("in-ar");
    runBtn.disabled = false;
    session = null;
    log("session ended");
  });

  await runArLoop(session);
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

  let lastDepthDraw = 0;
  let depthReported = false;
  let frames = 0;
  let debuggedViews = false;

  const onFrame: XRFrameRequestCallback = (time, frame) => {
    s.requestAnimationFrame(onFrame);
    frames++;

    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const pose = frame.getViewerPose(refSpace);
    if (!pose) return;

    if (!debuggedViews) {
      debuggedViews = true;
      log(`viewer pose ok, ${pose.views.length} view(s)`);
    }

    // --- hit test ---
    let hits = 0;
    if (hitTestSource) {
      hits = frame.getHitTestResults(hitTestSource).length;
    }
    reticleEl.classList.toggle("hit", hits > 0);
    reticleLabel.textContent = hits > 0 ? "SURFACE" : "";

    // --- depth ---
    let depthText = "depth: not exposed on this device";
    for (const view of pose.views) {
      const depth = frame.getDepthInformation?.(view) as XRDepthInformation | null;
      if (!depth) continue;
      const cpu = depth as unknown as {
        getDepthInMeters?: (x: number, y: number) => number;
      };
      let centre: number | undefined;
      if (typeof cpu.getDepthInMeters === "function") {
        try {
          centre = cpu.getDepthInMeters(depth.width / 2, depth.height / 2);
        } catch {
          /* out of range */
        }
      }
      report.depth = {
        active: true,
        width: depth.width,
        height: depth.height,
        usage: report.session.depthUsage ?? undefined,
        format: report.session.depthDataFormat ?? undefined,
        centreMeters: centre,
      };
      depthWrap.classList.remove("off");
      depthText =
        `depth: ${depth.width}x${depth.height}` +
        (centre !== undefined ? ` centre=${centre.toFixed(2)}m` : "");
      if (!depthReported) {
        depthReported = true;
        log(`depth-sensing ACTIVE: ${depth.width}x${depth.height}`);
      }
      if (time - lastDepthDraw > 250 && cpu.getDepthInMeters) {
        lastDepthDraw = time;
        drawDepthPreview(cpu.getDepthInMeters.bind(depth), depth.width, depth.height);
      }
    }

    setArLive(
      `frame ${frames}  views ${pose.views.length}\n` +
        `hit-test ${hits ? `LOCK (${hits})` : "searching"}\n` +
        depthText,
    );
  };

  s.requestAnimationFrame(onFrame);
  log("render loop running");
}

function drawDepthPreview(
  sample: (x: number, y: number) => number,
  srcW: number,
  srcH: number,
) {
  const ctx = depthCanvas.getContext("2d");
  if (!ctx) return;
  const w = depthCanvas.width;
  const h = depthCanvas.height;
  // Canvas is a fixed preview size; scale samples from the raw depth extent.
  const img = ctx.createImageData(srcW, srcH);
  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < srcW; x++) {
      let m = NaN;
      try {
        m = sample(x, y);
      } catch {
        /* ignore */
      }
      const v = Number.isFinite(m) ? Math.max(0, Math.min(1, 1 - m / 5)) : 0;
      const i = (y * srcW + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v * 255;
      img.data[i + 3] = 255;
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = srcW;
  tmp.height = srcH;
  tmp.getContext("2d")!.putImageData(img, 0, 0);
  ctx.drawImage(tmp, 0, 0, w, h);
}

// --- sending the report -----------------------------------------------------

async function sendReport(): Promise<boolean> {
  report.timestamp = new Date().toISOString();
  const text = JSON.stringify(report, null, 2);
  try {
    const res = await fetch(REPORT_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: text,
    });
    const result = (await res.json()) as { ok: boolean; saved?: string; error?: string };
    if (!res.ok || !result.ok) throw new Error(result.error ?? `HTTP ${res.status}`);
    log(`report saved: ${result.saved}`);
    setArLive(`${arLiveEl.textContent}\n\n✔ report saved to\n${result.saved}`);
    arSendBtn.textContent = "Sent ✔";
    return true;
  } catch (e) {
    log(`send failed: ${String(e)}`);
    // Fallback so the report is not lost if the dev-server sink is unavailable.
    try {
      await navigator.clipboard.writeText(text);
      arSendBtn.textContent = "Copied (send failed)";
    } catch {
      arSendBtn.textContent = "Send failed";
    }
    return false;
  }
}

// --- wiring -----------------------------------------------------------------

async function init() {
  arSendBtn.addEventListener("click", () => void sendReport());
  arExitBtn.addEventListener("click", () => void session?.end());
  runBtn.addEventListener("click", () => void startSessionTest());

  const canRun = await probeEnvironment();
  runBtn.disabled = !canRun;
  runBtn.textContent = canRun
    ? "Run AR capability test"
    : "AR not supported on this browser/device";
  if (!canRun) log("Cannot start immersive-ar session here.");
}

void init();
