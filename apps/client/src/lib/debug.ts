/**
 * Minimal on-screen debug log + error capture.
 *
 * Essential while we can't attach a debugger to the phone: any uncaught error
 * or unhandled rejection is shown in the corner of the page. Also keeps a ring
 * buffer that can be copied/sent by hand.
 */
const lines: string[] = [];
let el: HTMLElement | null = null;

function render() {
  if (!el) return;
  el.textContent = lines.slice(-12).join("\n");
}

export function installDebugOverlay(target?: HTMLElement | null) {
  el = target ?? document.getElementById("debug");
  if (!el) {
    el = document.createElement("pre");
    el.id = "debug";
    Object.assign(el.style, {
      position: "fixed",
      left: "8px",
      bottom: "8px",
      right: "8px",
      maxHeight: "28vh",
      overflow: "auto",
      margin: "0",
      padding: "6px 8px",
      fontSize: "11px",
      lineHeight: "1.35",
      color: "#9fe8cf",
      background: "rgba(0,0,0,.55)",
      borderRadius: "8px",
      pointerEvents: "none",
      whiteSpace: "pre-wrap",
      zIndex: "1000",
    } as CSSStyleDeclaration);
    document.body.appendChild(el);
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
