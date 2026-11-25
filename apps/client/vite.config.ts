import basicSsl from "@vitejs/plugin-basic-ssl";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// WebXR + camera need a *secure context*. On the desktop, http://localhost
// qualifies. On a phone hitting the LAN IP it does not - so the dev server
// serves HTTPS with a self-signed cert (basicSsl). The phone shows a warning
// once; tap "Advanced -> Proceed" and you are done. No cloudflared needed.
//
// Prefer no warning at all? Use USB port forwarding instead (Android):
//   adb reverse tcp:5173 tcp:5173   then open http://localhost:5173 on the phone.
export default defineConfig({
  plugins: [basicSsl()],
  server: {
    host: true, // bind 0.0.0.0 so the phone can reach it over LAN
    port: 5173,
    // Proxy API + WebSocket to the gateway so the HTTPS page never makes
    // mixed-content (https page -> ws://) requests. Client code should use
    // relative URLs: fetch("/api/...") and new WebSocket(`.../ws`).
    proxy: {
      "/api": { target: "http://localhost:3000", changeOrigin: true },
      "/ws": { target: "ws://localhost:3000", ws: true },
    },
  },
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        capabilities: fileURLToPath(
          new URL("./capabilities.html", import.meta.url),
        ),
      },
    },
  },
});
