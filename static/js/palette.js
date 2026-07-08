// Album-art accent extraction: downsample the artwork, find the dominant
// saturated hue, clamp its lightness for contrast against paper/ink, and set
// --accent on :root. Falls back to the stylesheet default on any failure
// (CORS, decode, all-grey art).

const SIZE = 16;
const DEFAULT_ACCENT = ""; // empty -> revert to the CSS default

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h * 360, s, l];
}

function extract(img) {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // throws if canvas is tainted

  // Score hue buckets by saturation and mid-lightness.
  const buckets = new Map();
  for (let i = 0; i < data.length; i += 4) {
    const [h, s, l] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
    if (s < 0.25 || l < 0.12 || l > 0.9) continue;
    const key = Math.round(h / 24) * 24;
    const weight = s * (1 - Math.abs(l - 0.5));
    const bucket = buckets.get(key) || { weight: 0, s: 0, n: 0 };
    bucket.weight += weight;
    bucket.s += s;
    bucket.n += 1;
    buckets.set(key, bucket);
  }
  if (!buckets.size) return null;

  let bestHue = null, bestWeight = -1, bestSat = 0;
  for (const [hue, b] of buckets) {
    if (b.weight > bestWeight) {
      bestWeight = b.weight;
      bestHue = hue;
      bestSat = b.s / b.n;
    }
  }
  return { hue: bestHue, sat: Math.min(0.85, Math.max(0.45, bestSat)) };
}

export function applyAccentFromImage(img) {
  let accent = DEFAULT_ACCENT;
  try {
    const result = extract(img);
    if (result) {
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      // Clamp lightness so the accent stays readable on paper (light) / ink (dark).
      const lightness = dark ? 62 : 38;
      accent = `hsl(${result.hue} ${Math.round(result.sat * 100)}% ${lightness}%)`;
    }
  } catch { /* tainted canvas or decode failure -> default accent */ }
  document.documentElement.style.setProperty("--accent", accent);
}

export function watchArtwork(imgEl) {
  imgEl.crossOrigin = "anonymous"; // i.scdn.co and mzstatic serve CORS headers
  imgEl.addEventListener("load", () => applyAccentFromImage(imgEl));
  imgEl.addEventListener("error", () =>
    document.documentElement.style.setProperty("--accent", DEFAULT_ACCENT));
}
