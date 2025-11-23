// Helpers for body map coordinate inference

// anchors: array of { name, leftPct, topPct }
export function inferBodyPartFromCoords(coords, anchors = []) {
  if (!coords || !anchors || anchors.length === 0) return null;
  const x = coords.leftPct;
  const y = coords.topPct;
  let best = null;
  let bestDist = Infinity;
  for (const a of anchors) {
    const dx = (a.leftPct || 0) - x;
    const dy = (a.topPct || 0) - y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < bestDist) {
      bestDist = dist;
      best = a;
    }
  }
  return best ? best.name : null;
}
