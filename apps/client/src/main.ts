/** Landing page: detects WebXR AR support and links to the two clients. */
import { installDebugOverlay, log } from "./lib/debug";

installDebugOverlay();
log("landing");

const arLink = document.getElementById("ar-link") as HTMLAnchorElement;
const note = document.getElementById("support-note")!;

(async () => {
  const xr = (navigator as Navigator & { xr?: XRSystem }).xr;
  const supported = xr ? await xr.isSessionSupported("immersive-ar") : false;
  if (supported) {
    note.textContent = "This device supports immersive-ar. Open the scan visor.";
  } else {
    arLink.classList.add("disabled");
    arLink.setAttribute("aria-disabled", "true");
    note.textContent =
      "immersive-ar not supported here. Use the desktop viewer, or open on Android Chrome.";
  }
})();
