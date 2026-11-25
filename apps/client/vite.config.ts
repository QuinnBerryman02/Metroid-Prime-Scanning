import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// WebXR + camera need a secure context. For phone testing, either:
//   - `mkcert` a local CA and provide certs (set VITE_HTTPS=1), or
//   - use a tunnel (cloudflared / ngrok) and point the phone at that URL.
function httpsConfig() {
  if (!process.env.VITE_HTTPS) return undefined;
  return {
    key: readFileSync(process.env.VITE_KEY ?? "./certs/key.pem"),
    cert: readFileSync(process.env.VITE_CERT ?? "./certs/cert.pem"),
  };
}

export default defineConfig({
  server: {
    host: true,
    port: 5173,
    https: httpsConfig(),
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
