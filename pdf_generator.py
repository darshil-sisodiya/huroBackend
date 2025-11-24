"""
PDF Health Report Generator
Generates professional medical reports from user health data.
"""

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional
import json


class NumberedCanvas(canvas.Canvas):
    """Custom canvas with page numbers and headers."""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColorRGB(0.5, 0.5, 0.5)
        self.drawRightString(
            7.5 * inch, 0.5 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )
        self.drawString(
            0.75 * inch, 0.5 * inch,
            f"Generated: {datetime.now().strftime('%B %d, %Y')}"
        )


def create_health_report_pdf(
    username: str,
    profile_data: Dict[str, Any],
    timeline_entries: List[Dict[str, Any]],
    insights_data: Dict[str, Any],
    ai_summary: str,
    health_score_data: Dict[str, Any]
) -> BytesIO:
    """
    Generate a comprehensive health report PDF.
    
    Args:
        username: Patient username
        profile_data: Health profile information
        timeline_entries: All timeline logs
        insights_data: Health insights and patterns
        ai_summary: AI-generated summary from Gemini
        health_score_data: Current health score breakdown
        
    Returns:
        BytesIO: PDF file buffer
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=1*inch,
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=rl_colors.HexColor('#1E293B'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=rl_colors.HexColor('#64748B'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=rl_colors.HexColor('#0F172A'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=rl_colors.HexColor('#334155'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    label_style = ParagraphStyle(
        'CustomLabel',
        parent=styles['Normal'],
        fontSize=9,
        textColor=rl_colors.HexColor('#64748B'),
        fontName='Helvetica-Bold'
    )
    
    # Title Page
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Health Report", title_style))
    story.append(Paragraph(f"Prepared for: {username}", subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report Info Box
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    info_data = [
        ['Report Date:', report_date],
        ['Patient:', username],
        ['Report Type:', 'Comprehensive Health Summary'],
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Health Score Section
    story.append(Paragraph("Overall Health Score", heading_style))
    
    score = health_score_data.get('score', 0)
    label = health_score_data.get('label', 'N/A')
    reason = health_score_data.get('reason', 'No data available')
    
    # Create score visualization
    score_color = rl_colors.HexColor('#10B981') if score >= 75 else rl_colors.HexColor('#F59E0B') if score >= 50 else rl_colors.HexColor('#EF4444')
    
    score_data = [
        [Paragraph(f"<font size=24 color='#{score_color.hexval()[2:]}'><b>{score}/100</b></font>", body_style),
         Paragraph(f"<b>Status:</b> {label}<br/><br/>{reason}", body_style)]
    ]
    
    score_table = Table(score_data, colWidths=[1.5*inch, 4.5*inch])
    score_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 1, rl_colors.HexColor('#CBD5E1')),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Health Profile Section
    story.append(Paragraph("Health Profile", heading_style))
    
    profile_items = []
    profile_labels = {
        'sleep_pattern': 'Sleep Pattern',
        'sleep_hours': 'Sleep Hours',
        'hydration_level': 'Hydration Level',
        'stress_level': 'Stress Level',
        'exercise_frequency': 'Exercise Frequency',
        'diet_type': 'Diet Type'
    }
    
    for key, label in profile_labels.items():
        value = profile_data.get(key, 'N/A')
        if isinstance(value, str):
            value = value.replace('_', ' ').title()
        elif isinstance(value, (int, float)):
            value = f"{value} hours" if 'hours' in key else str(value)
        profile_items.append([label + ':', str(value)])
    
    profile_table = Table(profile_items, colWidths=[2*inch, 4*inch])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(profile_table)
    
    # Page Break
    story.append(PageBreak())
    
    # Health Insights Section
    story.append(Paragraph("Health Patterns & Insights", heading_style))
    
    insights_items = [
        ['Total Health Entries (30 days):', str(insights_data.get('total_entries', 0))],
        ['Symptoms Logged:', str(insights_data.get('symptoms_this_month', 0))],
        ['Stress-Free Days:', str(insights_data.get('stress_free_days', 0))],
        ['Hydration Logs:', str(insights_data.get('hydration_logs', 0))],
        ['Symptom Trend:', insights_data.get('trends', {}).get('symptom_trend', 'N/A').replace('_', ' ').title()],
        ['Hydration Trend:', insights_data.get('trends', {}).get('hydration_trend', 'N/A').replace('_', ' ').title()],
    ]
    
    insights_table = Table(insights_items, colWidths=[2.5*inch, 3.5*inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(insights_table)
    story.append(Spacer(1, 0.2*inch))
    
    # AI Summary Section
    story.append(Paragraph("AI Health Analysis", heading_style))
    ai_summary_cleaned = ai_summary.replace('*', '').replace('#', '')
    story.append(Paragraph(ai_summary_cleaned, body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timeline Entries Section
    story.append(PageBreak())
    story.append(Paragraph("Detailed Health Log", heading_style))
    story.append(Paragraph(
        f"Complete record of all logged health entries ({len(timeline_entries)} total entries)",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Group entries by type
    entries_by_type = {}
    for entry in timeline_entries:
        entry_type = entry.get('entry_type', 'other')
        if entry_type not in entries_by_type:
            entries_by_type[entry_type] = []
        entries_by_type[entry_type].append(entry)
    
    # Display entries by type
    type_labels = {
        'symptom': 'Symptoms',
        'mood': 'Mood Logs',
        'medicine': 'Medications',
        'sleep': 'Sleep Records',
        'hydration': 'Hydration',
        'note': 'Notes'
    }
    
    for entry_type, entries in sorted(entries_by_type.items()):
        if not entries:
            continue
            
        story.append(Spacer(1, 0.15*inch))
        type_heading = ParagraphStyle(
            'TypeHeading',
            parent=heading_style,
            fontSize=13,
            textColor=rl_colors.HexColor('#475569'),
            spaceAfter=8,
            spaceBefore=10,
        )
        story.append(Paragraph(type_labels.get(entry_type, entry_type.title()), type_heading))
        
        # Create table for this type
        entry_rows = [['Date', 'Entry', 'Details']]
        
        for entry in sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]:  # Limit to 20 most recent
            timestamp = entry.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                try:
                    if isinstance(timestamp, datetime):
                        dt = timestamp
                    elif isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = None
                    
                    if dt:
                        date_str = dt.strftime('%m/%d/%Y')
                    else:
                        date_str = str(timestamp)[:10]
                except:
                    date_str = str(timestamp)[:10] if len(str(timestamp)) >= 10 else str(timestamp)
            else:
                date_str = 'N/A'
            
            title = entry.get('title', 'No title')
            
            # Build details
            details_parts = []
            if entry.get('description'):
                details_parts.append(entry['description'][:100])
            
            tags = entry.get('tags', [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except:
                    tags = []
            
            if tags:
                tag_str = ', '.join([f"{t}" for t in tags[:5]])
                details_parts.append(f"Tags: {tag_str}")
            
            if entry.get('severity'):
                details_parts.append(f"Severity: {entry['severity']}/5")
            
            details = ' | '.join(details_parts) if details_parts else '-'
            
            entry_rows.append([
                date_str,
                title[:40],
                details[:60]
            ])
        
        if len(entry_rows) > 1:
            entry_table = Table(entry_rows, colWidths=[1*inch, 2*inch, 3*inch])
            entry_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#E0E7FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.HexColor('#1E293B')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#CBD5E1')),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#F8FAFC')]),
            ]))
            story.append(entry_table)
        else:
            story.append(Paragraph("No entries recorded", body_style))
    
    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This health report is generated based on self-reported data and AI analysis. 
    It is intended for informational purposes only and should not be considered as 
    professional medical advice, diagnosis, or treatment. Always consult with a 
    qualified healthcare provider for medical advice and before making any decisions 
    about your health or treatment.
    
    The information in this report represents patterns and insights derived from 
    logged data and should be reviewed with your healthcare provider for accurate 
    medical interpretation.
    """
    story.append(Paragraph(disclaimer_text, body_style))
    
    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    buffer.seek(0)
    return buffer
