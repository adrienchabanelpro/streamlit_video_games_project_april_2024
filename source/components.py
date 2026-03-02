"""Reusable UI components for the modern Streamlit app.

Replaces streamlit-extras (colored_header, add_vertical_space) with
native Streamlit + custom HTML/CSS components.
"""

from __future__ import annotations

import streamlit as st
from config import ACCENT, BG_CARD, BORDER, SUCCESS, TEXT_MUTED


def metric_card(
    label: str,
    value: str | int | float,
    delta: str | None = None,
    icon: str | None = None,
    accent: str = ACCENT,
) -> None:
    """Display a styled metric card with left border accent."""
    icon_html = f'<span style="font-size:1.4rem;margin-right:8px">{icon}</span>' if icon else ""
    delta_html = ""
    if delta:
        delta_html = f'<div style="color:{SUCCESS};font-size:0.85rem;margin-top:4px">{delta}</div>'

    st.markdown(
        f"""
        <div style="
            background:{BG_CARD};
            border-left:4px solid {accent};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:12px;
        ">
            <div style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:4px">
                {icon_html}{label}
            </div>
            <div style="color:{accent};font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    """Render a clean section header with optional description."""
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def info_card(title: str, body: str, accent: str = ACCENT) -> None:
    """Display an information card with accent color."""
    st.markdown(
        f"""
        <div style="
            background:{BG_CARD};
            border:1px solid {BORDER};
            border-top:3px solid {accent};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:16px;
        ">
            <div style="font-weight:600;font-size:1rem;margin-bottom:8px;color:{accent}">
                {title}
            </div>
            <div style="font-size:0.9rem;color:#CBD5E1;line-height:1.6">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_card(
    name: str,
    description: str,
    row_count: str,
    fields: str,
    url: str | None = None,
    accent: str = ACCENT,
) -> None:
    """Display a data source card with metadata."""
    link_html = f'<a href="{url}" target="_blank" style="color:{accent}">{url}</a>' if url else ""
    st.markdown(
        f"""
        <div style="
            background:{BG_CARD};
            border:1px solid {BORDER};
            border-radius:12px;
            padding:20px;
            margin-bottom:16px;
        ">
            <div style="font-weight:700;font-size:1.1rem;color:{accent};margin-bottom:8px">
                {name}
            </div>
            <div style="color:#CBD5E1;font-size:0.9rem;line-height:1.6;margin-bottom:12px">
                {description}
            </div>
            <div style="display:flex;gap:24px;flex-wrap:wrap">
                <div>
                    <span style="color:{TEXT_MUTED};font-size:0.8rem">Lignes</span><br>
                    <span style="font-weight:600;color:#F1F5F9">{row_count}</span>
                </div>
                <div>
                    <span style="color:{TEXT_MUTED};font-size:0.8rem">Champs cles</span><br>
                    <span style="font-weight:600;color:#F1F5F9">{fields}</span>
                </div>
            </div>
            {f'<div style="margin-top:8px;font-size:0.8rem">{link_html}</div>' if link_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_step(
    number: int, title: str, description: str, accent: str = ACCENT
) -> None:
    """Display a pipeline step indicator."""
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:flex-start;
            gap:16px;
            margin-bottom:16px;
        ">
            <div style="
                min-width:36px;
                height:36px;
                border-radius:50%;
                background:{accent};
                color:#fff;
                display:flex;
                align-items:center;
                justify-content:center;
                font-weight:700;
                font-size:0.9rem;
            ">{number}</div>
            <div>
                <div style="font-weight:600;color:#F1F5F9;margin-bottom:4px">{title}</div>
                <div style="color:{TEXT_MUTED};font-size:0.85rem">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
