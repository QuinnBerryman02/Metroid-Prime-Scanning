/**
 * WebXR capability / depth-sensing diagnostic.
 *
 * Manual test page (reachable at /capabilities.html). See README.
 *
 * It requests an `immersive-ar` session with every feature we care about as
 * optional, reports which the runtime granted, then runs a minimal WebGL AR
 * loop that live-reports hit-test results and (if present) depth data.
 *
 * The report is POSTed to `/__diag/report`, which the Vite dev server writes to
 * `apps/client/reports/`. Global error handlers capture any exception so a
 * failed run still produces a useful report.
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
  live: Record<string, number | boolean | null>;
  errors: string[];
  log: string[];
}

// --- tiny DOM helpers -------------------------------------------------------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const runBtn = $<HTMLButtonElement>("runBtn");
const panelSendBtn = $<HTMLButtonElement>("panelSendBtn");
const arSendBtn = $<HTMLButtonElement>("arSendBtn");
const arExitBtn = $<HTMLButtonElement>("arExitBtn");
const sendStatus = $("sendStatus");
const envEl = $("env");
const featuresEl = $("features");
const sessionEl = $("session");
const errorsEl = $("errors");
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
  live: {},
  errors: [],
  log: [],
};

function recordError(msg: string) {
  report.errors.push(msg);
  errorsEl.innerHTML = report.errors
    .map((e) => `<div class="row"><span class="no">${e}</span></div>`)
    .join("");
  log(`ERROR: ${msg}`);
}

// Capture everything, including async rejections from the XR loop.
window.addEventListener("error", (e) =>
  recordError(`${e.message} @ ${e.filename}:${e.lineno}`),
);
window.addEventListener("unhandledrejection", (e) =>
  recordError(`unhandledrejection: ${String(e.reason)}`),
);

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
  report.timestamp = new Date().toISOString();
  report.errors = [];
  report.log = [];
  log("requesting immersive-ar session…");

  try {
    session = await xrSystem.requestSession("immersive-ar", {
      requiredFeatures: ["hit-test"],
      optionalFeatures: [...FEATURES],
      domOverlay: { root: $("overlay") },
    });
  } catch (e) {
    log(`requestSession failed: ${String(e)}`);
    featuresEl.innerHTML = row("session", `FAILED: ${String(e)}`, "no");
    runBtn.disabled = false;
    return;
  }

  const s = session;
  try {
    log(`session started (blend=${s.environmentBlendMode})`);
    document.body.classList.add("in-ar");

    const enabled =
      (s as XRSession & { enabledFeatures?: string[] }).enabledFeatures ?? null;
    report.enabledFeatures = enabled;
    log(`enabledFeatures = ${JSON.stringify(enabled)}`);
    renderFeatureTable(enabled);

    report.session = {
      interactionMode: s.interactionMode ?? null,
      environmentBlendMode: s.environmentBlendMode ?? null,
      depthUsage: (s as XRSession & { depthUsage?: string }).depthUsage ?? null,
      depthDataFormat:
        (s as XRSession & { depthDataFormat?: string }).depthDataFormat ?? null,
    };
    log(`session meta = ${JSON.stringify(report.session)}`);

    s.addEventListener("end", () => {
      document.body.classList.remove("in-ar");
      runBtn.disabled = false;
      session = null;
      log("session ended");
      void sendReport(); // auto-send on exit so nothing is lost
    });

    await runArLoop(s);
  } catch (e) {
    recordError(
      `startSessionTest: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`,
    );
    runBtn.disabled = false;
  }
}

function renderFeatureTable(enabled: string[] | null) {
  const enabledSet = new Set(enabled ?? []);
  const cls = (f: (typeof FEATURES)[number]) =>
    f === "hit-test" ? "yes" : !enabled ? "warn" : enabledSet.has(f) ? "yes" : "no";
  const label = (f: (typeof FEATURES)[number]) =>
    f === "hit-test" ? "yes (required)" : !enabled ? "?" : enabledSet.has(f) ? "yes" : "no";

  featuresEl.innerHTML = FEATURES.map((f) =>
    row(f, enabledSet.has(f) || f === "hit-test" ? "enabled" : enabled ? "unavailable" : "unknown", cls(f)),
  ).join("");

  arFeaturesEl.innerHTML = FEATURES.map(
    (f) =>
      `<div class="feat"><span class="name">${f}</span><span class="${cls(f)}">${label(f)}</span></div>`,
  ).join("");
}

async function runArLoop(s: XRSession) {
  log("runArLoop: creating WebGL context");
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", {
    xrCompatible: true,
    alpha: true,
    antialias: true,
  });
  if (!gl) {
    recordError("webgl2 unavailable - cannot start render loop");
    return;
  }
  await gl.makeXRCompatible();
  log("runArLoop: webgl compatible");

  const layer = new XRWebGLLayer(s, gl);
  s.updateRenderState({ baseLayer: layer, depthNear: 0.1, depthFar: 100 });

  let refSpace: XRReferenceSpace;
  try {
    refSpace = (await s.requestReferenceSpace("local")) as XRReferenceSpace;
    log("runArLoop: local reference space ok");
  } catch (e) {
    recordError(`requestReferenceSpace failed: ${String(e)}`);
    return;
  }

  let hitTestSource: XRHitTestSource | null = null;
  const requestHitTest = s.requestHitTestSource?.bind(s);
  if (requestHitTest) {
    try {
      hitTestSource = (await requestHitTest({ space: refSpace })) ?? null;
      log(`runArLoop: hit-test source ${hitTestSource ? "acquired" : "null"}`);
    } catch (e) {
      recordError(`requestHitTestSource failed: ${String(e)}`);
    }
  } else {
    recordError("requestHitTestSource not available on this session");
  }

  let frames = 0;
  let nullPoses = 0;
  let loopErrors = 0;

  const onFrame: XRFrameRequestCallback = (time, frame) => {
    s.requestAnimationFrame(onFrame);
    try {
      frames++;
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const pose = frame.getViewerPose(refSpace);
      if (!pose) {
        nullPoses++;
        report.live = { frames, nullPoses };
        setArLive(`frame ${frames}\nviewer pose: null (${nullPoses})`);
        return;
      }

      let hits = 0;
      if (hitTestSource) hits = frame.getHitTestResults(hitTestSource).length;
      reticleEl.classList.toggle("hit", hits > 0);
      reticleLabel.textContent = hits > 0 ? "SURFACE" : "";

      report.live = { frames, nullPoses, views: pose.views.length, hits };
      report.depth = report.depth.active ? report.depth : { active: false };
      setArLive(
        `frame ${frames}\nviews ${pose.views.length}\n` +
          `hit-test ${hits ? `LOCK (${hits})` : "searching"}\n` +
          `depth: unavailable on this device`,
      );

      if (frames === 1) log("render loop frame 1 ✓");
    } catch (e) {
      loopErrors++;
      if (loopErrors <= 3) {
        recordError(
          `onFrame: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`,
        );
      }
    }
  };

  s.requestAnimationFrame(onFrame);
  log("runArLoop: render loop scheduled");
}

// --- sending the report -----------------------------------------------------

async function sendReport(currentSession = true): Promise<boolean> {
  report.timestamp = new Date().toISOString();
  if (currentSession && session) {
    const l = report.live as Record<string, unknown>;
    if (typeof l.frames === "number") {
      report.log.push(`[send] live at send: ${JSON.stringify(l)}`);
    }
  }
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
    sendStatus.textContent = `Saved to apps/client/${result.saved}`;
    setArLive(`${arLiveEl.textContent}\n\n✔ report saved`);
    arSendBtn.textContent = "Sent ✔";
    return true;
  } catch (e) {
    recordError(`send failed: ${String(e)}`);
    try {
      await navigator.clipboard.writeText(text);
      sendStatus.textContent = "Send failed - copied to clipboard instead.";
    } catch {
      sendStatus.textContent = "Send failed and clipboard unavailable.";
    }
    return false;
  }
}

// --- wiring -----------------------------------------------------------------

async function init() {
  panelSendBtn.addEventListener("click", () => void sendReport());
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
