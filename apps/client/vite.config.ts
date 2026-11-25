import basicSsl from "@vitejs/plugin-basic-ssl";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const CLIENT_DIR = fileURLToPath(new URL(".", import.meta.url));
const REPORT_DIR = path.join(CLIENT_DIR, "reports");

/**
 * Tiny dev-only sink so the capability page can POST its report to the server
 * instead of relying on a manual copy/paste. Reports land in
 * `apps/client/reports/` (plus a stable `latest-capabilities.json`).
 */
function diagReportSink(): Plugin {
  const handler: import("node:http").RequestListener = (req, res) => {
    if (req.method !== "POST") {
      res.statusCode = 405;
      res.end();
      return;
    }
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        mkdirSync(REPORT_DIR, { recursive: true });
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const name = `capabilities-${stamp}.json`;
        writeFileSync(path.join(REPORT_DIR, name), body);
        writeFileSync(path.join(REPORT_DIR, "latest-capabilities.json"), body);
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ ok: true, saved: `reports/${name}` }));
      } catch (e) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ ok: false, error: String(e) }));
      }
    });
  };
  return {
    name: "diag-report-sink",
    configureServer(server) {
      server.middlewares.use("/__diag/report", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/__diag/report", handler);
    },
  };
}

// WebXR + camera need a *secure context*. On the desktop, http://localhost
// qualifies. On a phone hitting the LAN IP it does not - so the dev server
// serves HTTPS with a self-signed cert (basicSsl). The phone shows a warning
// once; tap "Advanced -> Proceed" and you are done. No cloudflared needed.
//
// Prefer no warning at all? Use USB port forwarding instead (Android):
//   adb reverse tcp:5173 tcp:5173   then open http://localhost:5173 on the phone.
export default defineConfig({
  plugins: [basicSsl(), diagReportSink()],
  server: {
    host: true, // bind 0.0.0.0 so the phone can reach it over LAN
    port: 5173,
    // Proxy API + WebSocket to the gateway so the HTTPS page never makes
    // mixed-content (https page -> ws://) requests. Client code should use
    // relative URLs: fetch("/api/...") and new WebSocket(`.../ws`).
    proxy: {
      "/api": {
        target: "http://localhost:3000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
      "/ws": { target: "ws://localhost:3000", ws: true },
    },
  },
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        ar: fileURLToPath(new URL("./ar.html", import.meta.url)),
        viewer: fileURLToPath(new URL("./viewer.html", import.meta.url)),
        scan: fileURLToPath(new URL("./scan.html", import.meta.url)),
        capabilities: fileURLToPath(
          new URL("./capabilities.html", import.meta.url),
        ),
      },
    },
  },
});
