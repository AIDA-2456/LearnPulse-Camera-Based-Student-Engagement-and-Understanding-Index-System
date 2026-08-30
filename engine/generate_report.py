"""
LearnPulse - teacher report generator.

Reads the engagement log produced by the analyser and produces a coherent PDF
report for the teacher, including data-driven recommended actions and guidance
on how the results should be interpreted and used.

Usage:
  python engine\\generate_report.py engagement_log_classroom_session.csv
  python engine\\generate_report.py engagement_log_classroom_session.csv --email teacher@university.edu

Requires:
  pip install reportlab
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, KeepTogether)

# ---------------- configuration ----------------
DROP_MARGIN = 15        # points below session average that counts as a drop
MIN_EVENT_SECONDS = 4   # ignore drops shorter than this
SMOOTH_SECONDS = 10

NAVY = colors.HexColor("#1E2761")
ACCENT = colors.HexColor("#3D6BD8")
GREEN = colors.HexColor("#1E8A5A")
AMBER = colors.HexColor("#B07A1E")
LIGHT = colors.HexColor("#F2F5FB")
GREY = colors.HexColor("#5F6A85")


def fmt(seconds):
    return f"{int(seconds)//60}m {int(seconds)%60:02d}s"


# ---------------- load data ----------------
if len(sys.argv) < 2:
    print("Usage: python engine\\generate_report.py <engagement_log.csv> [--email address]")
    sys.exit()

csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    print(f"ERROR: could not find {csv_path}")
    sys.exit()

email_to = None
if "--email" in sys.argv:
    i = sys.argv.index("--email")
    if i + 1 < len(sys.argv):
        email_to = sys.argv[i + 1]

df = pd.read_csv(csv_path)
session_name = (os.path.basename(csv_path)
                .replace("engagement_log_", "").replace(".csv", ""))
has_phones = "phones_visible" in df.columns

eng = df["engaged_percent"].astype(float).values
times = df["time_seconds"].astype(float).values
duration = times[-1] if len(times) else 0

k = max(1, min(SMOOTH_SECONDS, len(eng) // 5))
smoothed = np.convolve(eng, np.ones(k) / k, mode="same")

avg = float(np.mean(eng))
lowest_i = int(np.argmin(eng))
highest_i = int(np.argmax(smoothed))
threshold = avg - DROP_MARGIN

# ---------------- detect attention-drop events ----------------
events, start = [], None
low = smoothed < threshold
for i, flag in enumerate(low):
    if flag and start is None:
        start = times[i]
    elif not flag and start is not None:
        if times[i - 1] - start >= MIN_EVENT_SECONDS:
            seg = eng[(times >= start) & (times <= times[i - 1])]
            events.append({"start": start, "end": times[i - 1],
                           "low": float(seg.min())})
        start = None
if start is not None and times[-1] - start >= MIN_EVENT_SECONDS:
    seg = eng[times >= start]
    events.append({"start": start, "end": times[-1], "low": float(seg.min())})

# ---------------- phone statistics ----------------
phone_peak = phone_share = 0
if has_phones:
    ph = df["phones_visible"].astype(float).values
    phone_peak = int(ph.max())
    phone_share = float(np.mean(ph > 0)) * 100

# ---------------- chart ----------------
chart_file = f"report_chart_{session_name}.png"
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.plot(times / 60, eng, color="#C9D2E4", linewidth=1)
ax.plot(times / 60, smoothed, color="#1D9E75", linewidth=2.2)
ax.axhline(avg, color="#999999", linestyle="--", linewidth=1)
for ev in events:
    ax.axvspan(ev["start"] / 60, ev["end"] / 60, color="#E24B4A", alpha=0.15)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Class engagement (%)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(chart_file, dpi=140)
plt.close(fig)

# ---------------- data-driven recommendations ----------------
recs = []

if events:
    first = events[0]
    recs.append(
        f"<b>Review the material taught between {fmt(first['start'])} and "
        f"{fmt(first['end'])}.</b> Engagement fell to {first['low']:.0f}% during this "
        f"period, well below the session average of {avg:.0f}%. Consider whether the "
        f"concept introduced here was conceptually difficult, delivered too quickly, or "
        f"presented without a worked example, and plan to revisit it early in the next "
        f"session.")

late_events = [e for e in events if e["start"] > duration * 0.6]
if late_events:
    recs.append(
        "<b>Consider introducing a short break or an activity change in the final third "
        "of the session.</b> Attention declined most in the later part of the class, "
        "which is consistent with the natural attention decay observed in extended "
        "teaching sessions. A two-minute pause, a question to the room, or a switch to "
        "paired discussion typically restores attention more effectively than continuing "
        "at the same pace.")

if len(events) >= 3:
    recs.append(
        f"<b>Review the overall pacing of the session.</b> {len(events)} separate "
        f"attention drops were detected, which suggests the difficulty may lie in the "
        f"rhythm of delivery rather than in any single topic. Alternating between "
        f"exposition and short interactive segments every 10 to 15 minutes is a "
        f"well-established remedy.")

if has_phones and phone_share > 15:
    recs.append(
        f"<b>Phone use was visible for {phone_share:.0f}% of the session, peaking at "
        f"{phone_peak} devices at once.</b> Where this coincides with the attention drops "
        f"above, it is more likely a symptom than a cause. Rather than a prohibition, "
        f"consider a brief signposted phone break, or incorporating devices deliberately "
        f"through a live poll or lookup task.")

if avg >= 70 and not events:
    recs.append(
        f"<b>Engagement remained consistently high, averaging {avg:.0f}% with no "
        f"sustained drops detected.</b> It is worth noting what characterised this "
        f"session - the pacing, the balance of activity types, the material covered - so "
        f"that it can be repeated deliberately rather than by chance.")

if smoothed[highest_i] > avg + 8:
    recs.append(
        f"<b>Engagement peaked around {fmt(times[highest_i])}, reaching "
        f"{smoothed[highest_i]:.0f}%.</b> Identifying what was happening at this point is "
        f"as valuable as examining the low points: whatever format or material held "
        f"attention here is worth using more often.")

if not recs:
    recs.append(
        "<b>No sustained attention drops were detected in this session.</b> Engagement "
        "remained close to its average throughout, and no specific corrective action is "
        "indicated by these results.")

# ---------------- build the PDF ----------------
out_pdf = f"LearnPulse_Report_{session_name}.pdf"
doc = SimpleDocTemplate(out_pdf, pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title=f"LearnPulse Report - {session_name}")

ss = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=NAVY,
                    alignment=0, spaceAfter=2)
sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=10, textColor=GREY,
                     spaceAfter=10)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=NAVY,
                    spaceBefore=12, spaceAfter=6)
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=15,
                      spaceAfter=7)
small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8.5, leading=12,
                       textColor=GREY)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, spaceAfter=9)

story = []

story.append(Paragraph("Class Engagement Report", h1))
story.append(Paragraph(
    f"Session: {session_name} &nbsp;|&nbsp; Duration: {fmt(duration)} "
    f"&nbsp;|&nbsp; Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", sub))

# --- summary table ---
rows = [
    ["Average engagement", f"{avg:.0f}%"],
    ["Lowest point", f"{eng[lowest_i]:.0f}% at {fmt(times[lowest_i])}"],
    ["Attention drops detected", f"{len(events)}"],
]
if has_phones:
    rows.append(["Phone use", f"visible for {phone_share:.0f}% of session "
                              f"(peak {phone_peak} at once)"])

tbl = Table(rows, colWidths=[55 * mm, 100 * mm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
    ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
]))
story.append(tbl)

story.append(Paragraph("Engagement across the session", h2))
story.append(Image(chart_file, width=170 * mm, height=66 * mm))
story.append(Paragraph(
    "The green line shows class engagement smoothed over time; the dashed line marks the "
    "session average. Shaded areas indicate periods where engagement fell substantially "
    "below that average for a sustained interval.", small))

# --- flagged moments ---
story.append(Paragraph("Moments that warrant attention", h2))
if events:
    ev_rows = [["Period", "Lowest engagement"]]
    for ev in events:
        ev_rows.append([f"{fmt(ev['start'])} - {fmt(ev['end'])}", f"{ev['low']:.0f}%"])
    et = Table(ev_rows, colWidths=[80 * mm, 75 * mm])
    et.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
    ]))
    story.append(et)
else:
    story.append(Paragraph(
        "No sustained attention drops were detected during this session.", body))

# --- recommended actions ---
story.append(Paragraph("Recommended actions", h2))
story.append(Paragraph(
    "The following suggestions are derived from the patterns in this session's data. They "
    "are prompts for your own judgement rather than instructions: you know what was being "
    "taught at each of these moments, and that context should always take precedence over "
    "the measurement.", body))
for r in recs:
    story.append(Paragraph(f"&bull;&nbsp;&nbsp;{r}", bullet))

# --- how to use this report ---
story.append(Paragraph("How to use this report", h2))
story.append(Paragraph(
    "This report is intended as a private aid to your own reflection on the session. It is "
    "not an assessment of your teaching, and it is not a record of individual students: "
    "every figure it contains describes the class as a whole, and the source recording is "
    "deleted once the analysis is complete.", body))
story.append(Paragraph(
    "The most useful way to read it is to work backwards from the flagged moments. Take "
    "each timestamp in the table above and recall what was happening at that point in the "
    "class - which concept you were introducing, which format you were using, how long you "
    "had been speaking without interruption. The system can identify <i>when</i> attention "
    "fell, but only you can supply the <i>why</i>, and it is the combination of the two "
    "that turns a measurement into a teaching decision.", body))
story.append(Paragraph(
    "Three practical uses follow from this. In the short term, the flagged periods indicate "
    "material that may benefit from being revisited at the start of the next session. In "
    "the medium term, comparing reports across several sessions will show whether a change "
    "you have made - a different pacing, an added activity, an earlier break - has had a "
    "measurable effect. And in preparing future iterations of the same material, the "
    "periods of highest engagement are as informative as the lowest, since they indicate "
    "formats worth repeating deliberately.", body))
story.append(Paragraph(
    "A single report should be treated as one observation rather than a conclusion. "
    "Engagement varies with the time of day, the point in the term, the assessment "
    "calendar, and factors entirely outside the classroom. Patterns that persist across "
    "several sessions are meaningful; a single unusual session usually is not.", body))

# --- limitations ---
story.append(Paragraph("Interpreting the measurement", h2))
story.append(Paragraph(
    "This system measures <b>behavioural</b> engagement: the observable signals of "
    "attention such as head orientation, posture, stillness and visible device use. It "
    "does not measure understanding, interest, or any internal state, and it should not be "
    "read as doing so. A student may be attentive while appearing still, or thinking "
    "deeply while looking away; conversely, apparent attention does not guarantee "
    "comprehension.", body))
story.append(Paragraph(
    "Detection accuracy also varies with conditions. Students seated furthest from the "
    "camera, partially occluded, or working with their heads down are measured less "
    "reliably than those clearly in view, and fairness analysis has shown that accuracy is "
    "not perfectly uniform across individuals. This is the reason results are reported only "
    "at class level: the measurement is sufficiently reliable to describe the pattern of a "
    "room over time, and deliberately not used to characterise any individual student.", body))

story.append(Spacer(1, 10))
story.append(Paragraph(
    "Generated by LearnPulse - Camera-Based Student Engagement and Understanding Index "
    "System. Source video is not retained; this report contains class-level aggregate "
    "results only.", small))

doc.build(story)
print(f"Report written: {out_pdf}")

# ---------------- optional email ----------------
if email_to:
    print(f"\nTo email this report to {email_to}, configure SMTP below.")
    print("Set these environment variables first, then re-run:")
    print("  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS")
    host = os.environ.get("SMTP_HOST")
    if not host:
        print("SMTP_HOST not set - skipping send. The PDF has still been created.")
    else:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = f"Class engagement report - {session_name}"
        msg["From"] = os.environ.get("SMTP_USER", "learnpulse@localhost")
        msg["To"] = email_to
        msg.set_content(
            f"Your class engagement report for '{session_name}' is attached.\n\n"
            f"Average engagement: {avg:.0f}%\n"
            f"Attention drops detected: {len(events)}\n\n"
            "The report includes recommended actions and guidance on how to interpret "
            "the results. It describes the class as a whole; no individual student data "
            "is recorded, and the source recording has been deleted.\n\n"
            "LearnPulse")
        with open(out_pdf, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf",
                               filename=out_pdf)
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.send_message(msg)
        print(f"Emailed to {email_to}")
