// 30-second preview player with Spotify-embed fallback.

export class Player {
  constructor({ audioEl, playBtn, metaEl, volumeEl, embedWrap }) {
    this.audio = audioEl;
    this.playBtn = playBtn;
    this.metaEl = metaEl;
    this.embedWrap = embedWrap;

    this.audio.volume = volumeEl.value / 100;
    volumeEl.addEventListener("input", () => { this.audio.volume = volumeEl.value / 100; });

    this.playBtn.addEventListener("click", () => this.toggle());
    this.audio.addEventListener("play", () => this.#setPlaying(true));
    this.audio.addEventListener("pause", () => this.#setPlaying(false));
    this.audio.addEventListener("ended", () => this.#setPlaying(false));
    this.audio.addEventListener("error", () => {
      this.metaEl.textContent = "Preview playback failed";
      this.playBtn.disabled = true;
    });
  }

  #setPlaying(playing) {
    this.playBtn.textContent = playing ? "❚❚" : "▶";
    this.playBtn.setAttribute("aria-label", playing ? "Pause preview" : "Play preview");
  }

  toggle() {
    if (!this.audio.src) return;
    if (this.audio.paused) {
      this.audio.play().catch(() => { this.metaEl.textContent = "Playback not available"; });
    } else {
      this.audio.pause();
    }
  }

  load(track) {
    this.audio.pause();
    this.#setPlaying(false);

    if (track.preview_url) {
      this.audio.src = track.preview_url;
      this.audio.load();
      this.playBtn.disabled = false;
      this.metaEl.textContent = `30s preview — ${track.name}`;
      this.#hideEmbed();
    } else {
      this.audio.removeAttribute("src");
      this.playBtn.disabled = true;
      this.metaEl.textContent = "No preview — use the Spotify player:";
      this.#showEmbed(track.id);
    }
  }

  #showEmbed(trackId) {
    this.embedWrap.textContent = "";
    const iframe = document.createElement("iframe");
    iframe.src = `https://open.spotify.com/embed/track/${trackId}?utm_source=generator`;
    iframe.width = "100%";
    iframe.height = "80";
    iframe.allow = "autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture";
    iframe.loading = "lazy";
    this.embedWrap.appendChild(iframe);
    this.embedWrap.classList.remove("hidden");
  }

  #hideEmbed() {
    this.embedWrap.textContent = "";
    this.embedWrap.classList.add("hidden");
  }
}
