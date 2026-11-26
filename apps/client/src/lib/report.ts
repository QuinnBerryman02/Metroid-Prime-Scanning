/**
 * Automatic diagnostic reporting.
 *
 * Anything an un-cabled phone needs to tell us (errors, structured `info`, and
 * the on-screen log ring buffer) is collected and POSTed to the dev server's
 * `/__diag/report` sink, which writes it to `apps/client/reports/`. Auto-sends
 * on `pagehide` so a session that crashes on the device still leaves a file.
 */
import { debugText } from "./debug";

const REPORT_URL = "/__diag/report";

interface RecordedError {
  where: string;
  name: string;
  message: string;
  stack?: string;
  at: string;
}

export interface Diagnostics {
  /** Attach a structured key/value to the report. */
  info(key: string, value: unknown): void;
  /** Record an error (with stack when available). */
  error(where: string, e: unknown): void;
  /** POST the report now. Resolves true when the sink accepted it. */
  send(reason: string): Promise<boolean>;
  /** Send a beacon if the page is hidden before we sent anything. */
  installAutoSend(): void;
}

function describe(e: unknown): { name: string; message: string; stack?: string } {
  if (e instanceof Error) {
    return { name: e.name, message: e.message, stack: e.stack };
  }
  if (e && typeof e === "object") {
    const o = e as { name?: unknown; message?: unknown };
    return {
      name: typeof o.name === "string" ? o.name : "Error",
      message: typeof o.message === "string" ? o.message : safeString(e),
      stack: (e as { stack?: string }).stack,
    };
  }
  return { name: typeof e, message: String(e) };
}

function safeString(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function createDiagnostics(kind: string): Diagnostics {
  const info: Record<string, unknown> = {};
  const errors: RecordedError[] = [];
  let sentAt = 0;

  function record(where: string, e: unknown) {
    const d = describe(e);
    errors.push({ where, ...d, at: new Date().toISOString() });
    // eslint-disable-next-line no-console
    console.log(`[diag] ${where}: ${d.name}: ${d.message}`);
  }

  window.addEventListener("error", (ev) =>
    record("window.onerror", ev.error ?? ev.message),
  );
  window.addEventListener("unhandledrejection", (ev) =>
    record("unhandledrejection", ev.reason),
  );

  function build(reason: string): string {
    return JSON.stringify(
      {
        kind,
        reason,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        secureContext: window.isSecureContext,
        url: location.href,
        info,
        errors,
        log: debugText().split("\n"),
      },
      null,
      2,
    );
  }

  async function send(reason: string): Promise<boolean> {
    sentAt = Date.now();
    const body = build(reason);
    try {
      const res = await fetch(REPORT_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
      });
      const json = (await res.json()) as { ok?: boolean; saved?: string };
      // eslint-disable-next-line no-console
      console.log(`[diag] report sent (${reason}) -> ${json.saved ?? res.status}`);
      return Boolean(json.ok);
    } catch (e) {
      record("report.send", e);
      beacon(build(reason));
      return false;
    }
  }

  function beacon(body: string) {
    try {
      navigator.sendBeacon(REPORT_URL, new Blob([body], { type: "application/json" }));
    } catch {
      /* ignore */
    }
  }

  function installAutoSend() {
    window.addEventListener("pagehide", () => {
      if (Date.now() - sentAt < 2000) return; // just sent
      beacon(build("pagehide"));
    });
  }

  return {
    info: (key, value) => {
      info[key] = value;
    },
    error: record,
    send,
    installAutoSend,
  };
}
