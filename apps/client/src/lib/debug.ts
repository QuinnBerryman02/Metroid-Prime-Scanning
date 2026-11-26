/**
 * Minimal on-screen debug log + error capture.
 *
 * Essential while we can't attach a debugger to the phone: any uncaught error
 * or unhandled rejection is shown in the corner of the page. Also keeps a ring
 * buffer that can be copied/sent by hand.
 */
const lines: string[] = [];
let el: HTMLElement | null = null; // the <pre> holding the lines
let head: HTMLElement | null = null; // the <summary>, shows the newest line
let box: HTMLDetailsElement | null = null;

function render() {
  if (!el) return;
  el.textContent = lines.slice(-12).join("\n");
  if (head) {
    // Folded away by default: the panel is tall and full of timestamps, which
    // is exactly what you stop wanting once things work. Folded, it still
    // shows the newest line, so an error is never silent.
    const last = lines[lines.length - 1] ?? "log";
    head.textContent = box?.open ? "log \u25be" : last;
  }
}

export function installDebugOverlay(target?: HTMLElement | null) {
  el = target ?? null;
  if (!el && !document.getElementById("debug")) {
    // A <details> gets folding for free, and half the width so it stops
    // stretching across the scene. Colours come from the viewer's theme
    // variables where they exist. 
    box = document.createElement("details");
    box.id = "debug";
    box.style.cssText = [
      "position:fixed",
      "top:58px",
      "right:12px",
      "max-width:46ch",
      "max-height:34vh",
      "overflow:auto",
      "margin:0",
      "padding:6px 8px",
      "font:11px/1.35 ui-monospace, Menlo, monospace",
      "color:var(--on-panel,#9fe8cf)",
      "background:var(--panel,rgba(0,0,0,.6))",
      "border:1px solid var(--border,rgba(255,255,255,.16))",
      "border-radius:8px",
      "z-index:1000",
    ].join(";");

    head = document.createElement("summary");
    head.style.cssText =
      "cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    head.textContent = "log";

    el = document.createElement("pre");
    el.style.cssText = "margin:4px 0 0;white-space:pre-wrap";

    box.append(head, el);
    box.addEventListener("toggle", render);
    document.body.appendChild(box);
  }

  // Adopt an overlay a previous module instance already built. Without this a
  // hot reload re-runs the module, finds #debug present, leaves `el` null, and
  // logging silently stops - which is exactly the kind of quiet failure this
  // overlay exists to prevent.
  if (!el) {
    const existing = document.getElementById("debug");
    if (existing instanceof HTMLDetailsElement) {
      box = existing;
      head = existing.querySelector("summary");
      el = existing.querySelector("pre");
      box.addEventListener("toggle", render);
    }
  }

  window.addEventListener("error", (e) =>
    log(`ERR ${e.message} @${e.filename?.split("/").pop() ?? "?"}:${e.lineno}`),
  );
  window.addEventListener("unhandledrejection", (e) =>
    log(`REJ ${String((e as PromiseRejectionEvent).reason)}`),
  );
}

export function log(msg: string) {
  const t = new Date().toISOString().slice(11, 19);
  lines.push(`[${t}] ${msg}`);
  // eslint-disable-next-line no-console
  console.log("[ms]", msg);
  render();
}

export function debugText(): string {
  return lines.join("\n");
}
