# backend/report_generator.py
# Generates a nicely formatted PDF fraud analysis report using reportlab.

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

REPORT_PATH = os.path.join(os.path.dirname(__file__), "fraud_analysis_report.pdf")

# ── colour palette ──────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1e3a5f")
MID_BLUE  = colors.HexColor("#2d6a9f")
LIGHT_BG  = colors.HexColor("#f0f4f8")
RED_ALERT = colors.HexColor("#c0392b")
GREEN_OK  = colors.HexColor("#1e8449")
WHITE     = colors.white


def _styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=22,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#cce0f5"),
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    section = ParagraphStyle(
        "Section",
        parent=base["Heading2"],
        fontSize=13,
        textColor=DARK_BLUE,
        spaceBefore=14,
        spaceAfter=4,
        borderPad=2,
    )
    body = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2c3e50"),
    )
    kv = ParagraphStyle(
        "KV",
        parent=body,
        fontSize=9,
        textColor=colors.HexColor("#34495e"),
    )
    return title, subtitle, section, body, kv


def _header_table(title_text, subtitle_text, s_title, s_sub):
    """Dark-blue banner at the top of the report."""
    header_para = Paragraph(title_text, s_title)
    sub_para = Paragraph(subtitle_text, s_sub)
    tbl = Table([[header_para], [sub_para]], colWidths=[17 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("ROUNDEDCORNERS", [6]),
            ]
        )
    )
    return tbl


def _kv_table(rows, s_body):
    """Two-column key-value table for stats."""
    table_data = []
    for k, v in rows:
        table_data.append(
            [Paragraph(f"<b>{k}</b>", s_body), Paragraph(str(v), s_body)]
        )
    tbl = Table(table_data, colWidths=[7 * cm, 10 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def generate_report(dataset_info, greedy_result, dc_result, dp_result, bt_result, bb_result, predictions=None):
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    s_title, s_sub, s_section, s_body, s_kv = _styles()
    story = []

    # ── Cover banner ────────────────────────────────────────────────────────
    story.append(
        _header_table(
            "AI Financial Fraud Detection System",
            f"Automated Fraud Analysis Report  •  Generated {datetime.now().strftime('%d %B %Y, %H:%M')}",
            s_title,
            s_sub,
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # ── Dataset overview ────────────────────────────────────────────────────
    story.append(Paragraph("1. Dataset Overview", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    if "error" not in dataset_info:
        amt = dataset_info.get("amount_stats", {})
        kv_rows = [
            ("Total Transactions", f"{dataset_info['total_transactions']:,}"),
            ("Fraud Transactions", f"{dataset_info['fraud_transactions']:,}"),
            ("Legitimate Transactions", f"{dataset_info['legitimate_transactions']:,}"),
            ("Fraud Rate", f"{dataset_info['fraud_rate']}%"),
            ("Unique Senders", dataset_info["unique_senders"]),
            ("Unique Receivers", dataset_info["unique_receivers"]),
            ("Amount — Min", f"${amt.get('min', 0):,.2f}"),
            ("Amount — Max", f"${amt.get('max', 0):,.2f}"),
            ("Amount — Mean", f"${amt.get('mean', 0):,.2f}"),
            ("Amount — Std Dev", f"${amt.get('std', 0):,.2f}"),
        ]
        story.append(_kv_table(kv_rows, s_body))
    else:
        story.append(Paragraph("No dataset loaded.", s_body))

    story.append(Spacer(1, 0.4 * cm))

    # ── Greedy suspicious transactions ──────────────────────────────────────
    story.append(Paragraph("2. Greedy — Suspicious Transaction Detection", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    if "error" not in greedy_result:
        story.append(
            Paragraph(
                f"Threshold: <b>${greedy_result['threshold']:,.2f}</b> "
                f"(mean + 2σ).  "
                f"Suspicious transactions found: <b>{greedy_result['suspicious_count']}</b>",
                s_body,
            )
        )
        story.append(Spacer(1, 0.2 * cm))

        top = greedy_result.get("top_suspicious", [])
        if top:
            header = ["Sender", "Receiver", "Amount", "Method", "Fraud"]
            rows = [header] + [
                [
                    t["sender"][:18],
                    t["receiver"][:18],
                    f"${t['amount']:,.2f}",
                    t["payment_method"],
                    "YES" if t["fraud_flag"] else "NO",
                ]
                for t in top[:12]
            ]
            tbl = Table(rows, colWidths=[3.5*cm, 3.5*cm, 3*cm, 3*cm, 2*cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("ROWBACKGROUNDS", (1, 1), (-1, -1), [WHITE, LIGHT_BG]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                        ("TEXTCOLOR", (4, 1), (4, -1), RED_ALERT),
                    ]
                )
            )
            story.append(tbl)

    story.append(Spacer(1, 0.4 * cm))

    # ── Divide & Conquer ────────────────────────────────────────────────────
    story.append(Paragraph("3. Divide and Conquer — Sorting Analysis", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    story.append(
        Paragraph(
            f"Processed <b>{dc_result.get('total_processed', 0)}</b> transactions using "
            f"Merge Sort and Quick Sort.",
            s_body,
        )
    )
    story.append(
        Paragraph(
            f"Top sorted values (Merge Sort): "
            f"{', '.join(str(x) for x in dc_result.get('merge_sorted_sample', [])[:8])}",
            s_body,
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # ── Dynamic Programming ─────────────────────────────────────────────────
    story.append(Paragraph("4. Dynamic Programming — Transaction Chain Analysis", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    ms = dp_result.get("max_subarray", {})
    lis = dp_result.get("longest_increasing_subsequence", {})
    story.append(
        Paragraph(
            f"Maximum burst window total: <b>${ms.get('sum', 0):,.2f}</b> "
            f"(indices {ms.get('start_index', 0)}–{ms.get('end_index', 0)}).",
            s_body,
        )
    )
    story.append(
        Paragraph(
            f"Longest escalating amount sequence length: <b>{lis.get('length', 0)}</b>.",
            s_body,
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # ── Backtracking ────────────────────────────────────────────────────────
    story.append(Paragraph("5. Backtracking — Fraud Pattern Combinations", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    story.append(
        Paragraph(
            f"Threshold: <b>${bt_result.get('threshold', 0):,.2f}</b>.  "
            f"Combinations found: <b>{bt_result.get('combinations_found', 0)}</b>.",
            s_body,
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # ── Branch & Bound ──────────────────────────────────────────────────────
    story.append(Paragraph("6. Branch and Bound — Optimised Selection", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=6))

    story.append(
        Paragraph(
            f"Optimal total from {bb_result.get('selected_count', 0)} selected transactions: "
            f"<b>${bb_result.get('optimal_total', 0):,.2f}</b> "
            f"(capacity = {bb_result.get('capacity', 0)}).",
            s_body,
        )
    )

    story.append(Spacer(1, 0.6 * cm))

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_tbl = Table(
        [[Paragraph("Confidential — AI Fraud Detection System  •  ADSA Project", s_kv)]],
        colWidths=[17 * cm],
    )
    footer_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(footer_tbl)

    doc.build(story)
    return REPORT_PATH
