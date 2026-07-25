// Scroll-driven page navigation: the wheel doesn't scroll the document, it
// scrubs between neighbouring views in the deck (ScrollTrigger-style scrub,
// fullPage-style snap on release). Input is hijacked only when the next page
// is unlocked — scrolling down needs canAdvance(current), scrolling up needs
// the document to be at its top. Buttons and keyboard keep using the
// one-shot View Transition path.
//
// Wheel notches arrive as discrete ~100px jumps, so raw input is choppy.
// Input only moves `target`; a rAF loop drives the rendered `progress`
// toward it with a critically damped spring. Unlike a plain lerp, the
// spring carries velocity across notches, so a burst of discrete inputs
// reads as one continuous glide instead of lurch-slow-lurch.

const SCRUB_DISTANCE = 1400; // px of wheel delta for a full transition
const STIFFNESS = 9;         // spring rad/s — lower = floatier, higher = tighter
const IDLE_MS = 140;         // gesture considered finished after this quiet gap
const SNAP_THRESHOLD = 0.12;  // travel needed before release commits to the
                             // next page; less springs back to the origin —
                             // a single wheel notch (~0.07) never commits

export function initScrollScrub({
  page, views, coverWordmark, seedWordmark,
  canAdvance, isBusy, onSettled,
}) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let pair = 0;      // scrubbing between views[pair] and views[pair + 1]
  let origin = 0;    // which end the gesture started from (0 or 1)
  let target = 0;    // where the input wants to be (0 = lower, 1 = upper)
  let progress = 0;  // what is rendered; springs toward target each frame
  let velocity = 0;  // progress units per second — continuity across notches
  let active = false;
  let snapping = false;
  let pairClass = "";
  let lastDir = 1;
  let idleTimer = 0;
  let frame = 0;
  let lastTime = 0;

  const current = () => views.findIndex((v) => !v.classList.contains("hidden"));

  function render() {
    page.style.setProperty("--p", String(progress));
    document.documentElement.style.setProperty("--deck", String(pair + progress));
  }

  function loop(now) {
    const dt = Math.min(0.064, (now - lastTime) / 1000);
    lastTime = now;
    // Critically damped spring (semi-implicit Euler): never overshoots
    // from rest, and velocity is continuous through target jumps.
    velocity += (STIFFNESS * STIFFNESS * (target - progress) - 2 * STIFFNESS * velocity) * dt;
    progress += velocity * dt;
    if (progress <= 0) { progress = 0; velocity = Math.max(0, velocity); }
    if (progress >= 1) { progress = 1; velocity = Math.min(0, velocity); }
    if (snapping && Math.abs(target - progress) < 0.002 && Math.abs(velocity) < 0.05) {
      progress = target;
      render();
      finalize(target);
      return;
    }
    render();
    frame = requestAnimationFrame(loop);
  }

  function begin(fromIndex, dir) {
    pair = dir > 0 ? fromIndex : fromIndex - 1;
    origin = target = progress = dir > 0 ? 0 : 1;
    velocity = 0;
    active = true;
    snapping = false;
    views[pair].classList.remove("hidden");
    views[pair + 1].classList.remove("hidden");
    pairClass = `scrub-${pair}${pair + 1}`;
    page.classList.add("scrubbing", pairClass);
    if (pair === 0) {
      // FLIP geometry: where the cover wordmark must land to become the
      // seed page-head wordmark. Measured fresh each gesture — the
      // viewport may have changed since the last one.
      const a = coverWordmark.getBoundingClientRect();
      const b = seedWordmark.getBoundingClientRect();
      page.style.setProperty("--dx", `${b.left - a.left}px`);
      page.style.setProperty("--dy", `${b.top - a.top}px`);
      page.style.setProperty("--s", String(b.height / a.height));
    }
    render();
    lastTime = performance.now();
    frame = requestAnimationFrame(loop);
  }

  function finalize(t) {
    const finalIndex = pair + t;
    active = false;
    cancelAnimationFrame(frame);
    clearTimeout(idleTimer);
    page.classList.remove("scrubbing", pairClass);
    for (const prop of ["--p", "--dx", "--dy", "--s"]) page.style.removeProperty(prop);
    document.documentElement.style.setProperty("--deck", String(finalIndex));
    views.forEach((v, i) => v.classList.toggle("hidden", i !== finalIndex));
    window.scrollTo(0, 0);
    onSettled(finalIndex);
  }

  function snap() {
    const far = 1 - origin;
    const movingAway = far === 1 ? lastDir > 0 : lastDir < 0;
    target = movingAway && Math.abs(target - origin) >= SNAP_THRESHOLD ? far : origin;
    snapping = true;
  }

  function nudge(delta) {
    if (delta !== 0) lastDir = delta > 0 ? 1 : -1;
    target = Math.min(1, Math.max(0, target + delta / SCRUB_DISTANCE));
    clearTimeout(idleTimer);
    if (target === 0 || target === 1) {
      snapping = true; // land smoothly at the boundary
      return;
    }
    snapping = false;
    idleTimer = setTimeout(snap, IDLE_MS);
  }

  function wantsHijack(delta, eventTarget) {
    if (reducedMotion.matches || isBusy()) return false;
    // Let the suggestions dropdown keep its own scroll.
    if (eventTarget instanceof Element && eventTarget.closest(".suggestions")) return false;
    const cur = current();
    if (cur < 0) return false;
    if (delta > 0) return cur < views.length - 1 && canAdvance(cur);
    return cur > 0 && window.scrollY <= 0;
  }

  function onWheel(e) {
    if (active) {
      e.preventDefault();
      nudge(e.deltaY);
      return;
    }
    if (!wantsHijack(e.deltaY, e.target)) return;
    e.preventDefault();
    begin(current(), e.deltaY > 0 ? 1 : -1);
    nudge(e.deltaY);
  }

  // Touch: vertical drags scrub the same progress value.
  let touchY = null;

  function onTouchStart(e) {
    touchY = e.touches[0].clientY;
  }

  function onTouchMove(e) {
    if (touchY == null) return;
    const dy = touchY - e.touches[0].clientY; // >0 = swiping up = "scroll down"
    if (!active && !wantsHijack(dy, e.target)) {
      touchY = e.touches[0].clientY;
      return;
    }
    e.preventDefault();
    if (!active) begin(current(), dy > 0 ? 1 : -1);
    touchY = e.touches[0].clientY;
    nudge(dy * 3); // finger travel is much shorter than wheel travel
  }

  function onTouchEnd() {
    touchY = null;
    if (active && !snapping) {
      clearTimeout(idleTimer);
      snap();
    }
  }

  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("touchstart", onTouchStart, { passive: true });
  window.addEventListener("touchmove", onTouchMove, { passive: false });
  window.addEventListener("touchend", onTouchEnd, { passive: true });

  return { isActive: () => active };
}
