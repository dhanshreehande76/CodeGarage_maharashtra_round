const API_URL = "http://127.0.0.1:8000";

export async function analyzeCase({ image, video, audio, claim }) {
  const form = new FormData();
  if (image) form.append("image", image);
  if (video) form.append("video", video);
  if (audio) form.append("audio", audio);
  if (claim && claim.trim()) form.append("claim", claim.trim());

  const res = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Analysis failed (HTTP ${res.status})`);
  }
  return res.json();
}