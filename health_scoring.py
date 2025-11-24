from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ScoreBreakdown:
  score: int
  label: str
  reason: str


NEGATIVE_MOODS = {"Stressed", "Anxious", "Low energy", "Sad", "Angry"}
POSITIVE_MOODS = {"Calm", "Happy", "Grateful", "Proud"}


def _parse_tags(raw: Any) -> Dict[str, List[str]]:
  grouped: Dict[str, List[str]] = {}
  if not raw:
    return grouped
  if isinstance(raw, str):
    import json
    try:
      raw = json.loads(raw)
    except Exception:
      raw = []
  if not isinstance(raw, Iterable):
    return grouped
  for item in raw:
    try:
      s = str(item)
      if ":" in s:
        key, value = s.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        grouped.setdefault(key, []).append(value)
    except Exception:
      continue
  return grouped


def _clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def compute_health_score(entries: List[Dict[str, Any]], now: Optional[datetime] = None) -> ScoreBreakdown:
  """Compute a deterministic health score from recent timeline entries.

  Uses only timeline data (types + tags) with no AI. The goal is to be
  strict but explainable, so we combine several dimensions:
  - mood balance and intensity
  - sleep duration and quality
  - hydration volume and consistency
  - symptom frequency and recency
  - logging consistency over the last 7 days
  """
  if now is None:
    now = datetime.utcnow()

  if not entries:
    return ScoreBreakdown(
      score=72,
      label="Not enough data",
      reason="You don't have many recent logs yet, so this is a cautious estimate.",
    )

  seven_days_ago = now - timedelta(days=7)
  recent = [e for e in entries if isinstance(e.get("timestamp"), datetime) and e["timestamp"] >= seven_days_ago]
  if not recent:
    recent = entries

  base_score = 75.0
  mood_score = 0.0
  sleep_score = 0.0
  hydration_score = 0.0
  symptom_score = 0.0
  consistency_score = 0.0

  mood_points = 0.0
  mood_count = 0
  negative_mood_days = set()
  positive_mood_days = set()

  total_sleep_hours = 0.0
  sleep_nights = 0
  restless_nights = 0

  total_cups = 0
  hydration_logs = 0

  symptom_logs = 0
  recent_severe_symptoms = 0

  days_with_logs = set()

  for e in recent:
    ts = e.get("timestamp")
    if isinstance(ts, datetime):
      days_with_logs.add(ts.date())
    entry_type = e.get("entry_type") or ""
    tags = _parse_tags(e.get("tags"))

    if entry_type == "mood":
      mood_count += 1
      moods = tags.get("mood", [])
      intensities = tags.get("intensity", [])
      mood_value = 0.0
      for m in moods:
        if m in POSITIVE_MOODS:
          mood_value += 1.0
          positive_mood_days.add(ts.date()) if isinstance(ts, datetime) else None
        if m in NEGATIVE_MOODS:
          mood_value -= 1.5
          negative_mood_days.add(ts.date()) if isinstance(ts, datetime) else None
      for inten in intensities:
        if inten == "high":
          mood_value *= 1.4
        elif inten == "medium":
          mood_value *= 1.2
        elif inten == "low":
          mood_value *= 1.0
      mood_points += mood_value

    if entry_type == "sleep":
      hours_tags = tags.get("sleep", [])
      for h in hours_tags:
        val = h.rstrip("hH+")
        try:
          h_val = float(val)
        except Exception:
          h_val = 0.0
        if h_val > 0:
          total_sleep_hours += h_val
          sleep_nights += 1
      for q in tags.get("quality", []):
        ql = q.lower()
        if ql == "great":
          sleep_score += 2.0
        elif ql == "ok":
          sleep_score += 0.5
        elif ql == "restless":
          sleep_score -= 2.5
          restless_nights += 1

    if entry_type == "hydration":
      for c in tags.get("cups", []):
        try:
          cups_i = int(str(c))
        except Exception:
          cups_i = 0
        if cups_i > 0:
          total_cups += cups_i
          hydration_logs += 1

    if entry_type == "symptom":
      symptom_logs += 1
      severity = e.get("severity")
      try:
        sev = int(severity) if severity is not None else None
      except Exception:
        sev = None
      if sev is not None and sev >= 4:
        recent_severe_symptoms += 1

  if mood_count > 0:
    avg_mood = mood_points / mood_count
    mood_score = _clamp(avg_mood * 6.0, -20.0, 15.0)

  if sleep_nights > 0:
    avg_sleep = total_sleep_hours / sleep_nights
    if avg_sleep < 5.0:
      sleep_score -= 12.0
    elif avg_sleep < 6.5:
      sleep_score -= 6.0
    elif 7.0 <= avg_sleep <= 8.5:
      sleep_score += 8.0
    elif avg_sleep > 9.5:
      sleep_score -= 4.0
    if restless_nights >= 2:
      sleep_score -= 4.0
    sleep_score = _clamp(sleep_score, -20.0, 15.0)

  if hydration_logs > 0:
    avg_cups = total_cups / hydration_logs
    if avg_cups < 4:
      hydration_score -= 6.0
    elif avg_cups < 6:
      hydration_score -= 2.0
    elif 6 <= avg_cups <= 9:
      hydration_score += 6.0
    elif avg_cups > 12:
      hydration_score -= 3.0
    hydration_score = _clamp(hydration_score, -12.0, 10.0)

  if symptom_logs > 0:
    symptom_score -= min(symptom_logs * 2.0, 18.0)
  if recent_severe_symptoms > 0:
    symptom_score -= min(recent_severe_symptoms * 4.0, 20.0)
  symptom_score = _clamp(symptom_score, -30.0, 0.0)

  days_tracked = len(days_with_logs)
  if days_tracked == 0:
    consistency_score -= 6.0
  else:
    consistency_ratio = days_tracked / 7.0
    if consistency_ratio >= 0.8:
      consistency_score += 6.0
    elif consistency_ratio >= 0.5:
      consistency_score += 2.0
    else:
      consistency_score -= 4.0

  raw_score = base_score + mood_score + sleep_score + hydration_score + symptom_score + consistency_score
  final_score = int(round(_clamp(raw_score, 20.0, 95.0)))

  reasons: List[str] = []
  if negative_mood_days:
    reasons.append("several days with stressed or low moods")
  if positive_mood_days:
    reasons.append("some days with calm or happy moods")
  if sleep_nights == 0:
    reasons.append("little or no recent sleep tracking")
  else:
    if total_sleep_hours / max(sleep_nights, 1) < 6.0:
      reasons.append("short average sleep duration")
    if restless_nights >= 2:
      reasons.append("multiple restless nights")
  if hydration_logs == 0:
    reasons.append("no hydration logs")
  else:
    avg_cups = total_cups / max(hydration_logs, 1)
    if avg_cups < 6:
      reasons.append("low average water intake")
    elif 6 <= avg_cups <= 9:
      reasons.append("solid hydration most days")
  if symptom_logs > 0:
    reasons.append("recent symptoms logged")
  if recent_severe_symptoms > 0:
    reasons.append("some symptoms were marked as more severe")
  if days_tracked < 4:
    reasons.append("limited logging days this week")
  elif days_tracked >= 6:
    reasons.append("consistent logging across the week")

  if final_score >= 80:
    label = "Strong week"
  elif final_score >= 70:
    label = "Fairly balanced"
  elif final_score >= 55:
    label = "Mixed signals"
  elif final_score >= 40:
    label = "Needs support"
  else:
    label = "Rough patch"

  if not reasons:
    reason_text = "Based on your recent logs across mood, sleep, hydration and symptoms."
  else:
    reason_text = "This reflects " + ", ".join(reasons) + "."

  return ScoreBreakdown(score=final_score, label=label, reason=reason_text)
