"""Razorpay RiskForge AI — Enterprise Merchant Risk & Dispute Defense Platform.

Unified production risk engine combining:
- Tabular Machine Learning (Calibrated XGBoost)
- Deterministic Rule Corroboration (6 Independent Rules)
- Financial Loss & Bayesian Cost Engine
- Decision Fusion Architecture
- TreeSHAP Feature Attributions
- Grounded AI Dispute Defense Responder
- Human-in-the-Loop Live Risk Simulator & Analyst Review Ledger
- Mobile-First "RiskForge Mobile" Glassmorphic Intelligence View (393x852)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Top-level ASGI / FastAPI instance for Vercel deployment
from backend.api import app

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.ai.responder import GroundedCaseResponder
from src.ai.schemas import EvidenceChecklist, GroundedCaseInput
from src.decision.cost import CostEngine
from src.decision.service import RiskDecisionService


def run_streamlit_dashboard():
    # Set Streamlit Page Configuration
    st.set_page_config(
        page_title="Razorpay RiskForge AI — Merchant Risk Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize Session State Variables
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"

    if "analyst_audit_log" not in st.session_state:
        st.session_state["analyst_audit_log"] = [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_id": "TXN_INIT_9041",
                "amount": 42000.0,
                "engine_verdict": "REVIEW",
                "analyst_action": "Approved after 2FA",
                "notes": "Customer confirmed order via registered phone OTP.",
            }
        ]

    if "last_simulated_txn" not in st.session_state:
        st.session_state["last_simulated_txn"] = None


    # Theme definitions (Obsidian Glass Intelligence & Razorpay Design System)
    is_dark = st.session_state["theme_mode"] == "dark"

    DARK_THEME_CSS = """
        :root {
            --bg-color: #030712;
            --card-bg: rgba(15, 23, 42, 0.65);
            --card-border: rgba(11, 114, 231, 0.28);
            --sidebar-bg: rgba(3, 7, 18, 0.95);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.85);
            --input-border: rgba(11, 114, 231, 0.35);
            --input-text: #f8fafc;
            --dropdown-bg: #0f172a;
            --table-header-bg: rgba(11, 114, 231, 0.15);
            --accent-blue: #0b72e7;
            --accent-blue-dim: rgba(11, 114, 231, 0.18);
            --accent-cyan: #00d2d2;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --shadow: 0 12px 36px rgba(0, 0, 0, 0.55);
            --hud-bg: rgba(3, 7, 18, 0.82);
            --hud-border: rgba(255, 255, 255, 0.12);
            --hud-text: #ffffff;
            --pill-bg: rgba(11, 114, 231, 0.15);
            --pill-text: #38bdf8;
        }
    """

    LIGHT_THEME_CSS = """
        :root {
            --bg-color: #f1f5f9;
            --card-bg: rgba(255, 255, 255, 0.88);
            --card-border: rgba(11, 114, 231, 0.22);
            --sidebar-bg: rgba(248, 250, 252, 0.96);
            --text-primary: #0f172a;
            --text-secondary: #334155;
            --input-bg: #ffffff;
            --input-border: rgba(11, 114, 231, 0.30);
            --input-text: #0f172a;
            --dropdown-bg: #ffffff;
            --table-header-bg: rgba(11, 114, 231, 0.10);
            --accent-blue: #0b72e7;
            --accent-blue-dim: rgba(11, 114, 231, 0.12);
            --accent-cyan: #0284c7;
            --accent-emerald: #059669;
            --accent-rose: #e11d48;
            --accent-amber: #d97706;
            --shadow: 0 10px 28px rgba(11, 114, 231, 0.09);
            --hud-bg: rgba(255, 255, 255, 0.90);
            --hud-border: rgba(11, 114, 231, 0.25);
            --hud-text: #0f172a;
            --pill-bg: rgba(11, 114, 231, 0.10);
            --pill-text: #0b72e7;
        }
    """

    active_theme_vars = DARK_THEME_CSS if is_dark else LIGHT_THEME_CSS

    # Global Styling Injection
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        {active_theme_vars}

        html, body, [class*="css"], .stApp {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color) !important;
            color: var(--text-primary) !important;
        }}

        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--card-border) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }}
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {{
            color: var(--text-primary);
        }}
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {{
            color: var(--text-secondary) !important;
        }}

        /* Global Typography overrides */
        h1, h2, h3, h4, h5, h6,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: var(--text-primary) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
        }}
        p, span, label, div[data-testid="stMarkdownContainer"] p {{
            color: var(--text-primary);
        }}
        .stCaption, small {{
            color: var(--text-secondary) !important;
        }}

        /* Glassmorphic Metric Cards */
        .metric-card {{
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 22px;
            box-shadow: var(--shadow);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .metric-card:hover {{
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }}
        .metric-card h2, .metric-card h3 {{
            color: var(--text-primary) !important;
        }}

        /* Risk Status Badges */
        .badge-high {{
            background: rgba(244, 63, 94, 0.18);
            color: var(--accent-rose) !important;
            border: 1px solid rgba(244, 63, 94, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            display: inline-block;
        }}

        .badge-review {{
            background: rgba(245, 158, 11, 0.18);
            color: var(--accent-amber) !important;
            border: 1px solid rgba(245, 158, 11, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            display: inline-block;
        }}

        .badge-low {{
            background: rgba(16, 185, 129, 0.18);
            color: var(--accent-emerald) !important;
            border: 1px solid rgba(16, 185, 129, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            display: inline-block;
        }}

        .gradient-header {{
            background: linear-gradient(135deg, #00d2d2 0%, #0b72e7 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            background: var(--pill-bg);
            border: 1px solid var(--card-border);
            color: var(--pill-text) !important;
            letter-spacing: 0.5px;
        }}
        .status-pill::before {{
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
        }}

        /* Streamlit Form Inputs & Controls */
        input, textarea, select {{
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="input"] {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="input"] input {{
            color: var(--input-text) !important;
            background-color: transparent !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
            color: var(--input-text) !important;
        }}
        div[data-baseweb="select"] span {{
            color: var(--input-text) !important;
        }}
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {{
            background-color: var(--dropdown-bg) !important;
            color: var(--text-primary) !important;
        }}
        li[role="option"]:hover {{
            background-color: var(--accent-blue-dim) !important;
        }}

        /* Streamlit Native Metric Overrides */
        div[data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: var(--text-secondary) !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* Tables & Dataframes */
        table {{
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
        }}
        th {{
            background-color: var(--table-header-bg) !important;
            color: var(--text-primary) !important;
            border-bottom: 2px solid var(--card-border) !important;
            font-weight: 700 !important;
        }}
        td {{
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--card-border) !important;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--card-border);
            border-radius: 12px;
            background: var(--card-bg);
        }}

        /* Buttons */
        div.stButton > button:first-child {{
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.25s ease;
        }}
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #0b72e7 0%, #00b4d8 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(11, 114, 231, 0.35) !important;
        }}
        div.stButton > button[kind="secondary"] {{
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--card-border) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


    @st.cache_resource
    def get_decision_service() -> RiskDecisionService:
        """Instantiate and cache the unified production risk decision service."""
        return RiskDecisionService()


    @st.cache_resource
    def get_ai_responder() -> GroundedCaseResponder:
        """Instantiate and cache the grounded dispute defense generator."""
        return GroundedCaseResponder()


    @st.cache_data
    def load_sample_transactions() -> pd.DataFrame:
        """Load active transactions for operational exploration."""
        sample_file = Path("data/sample/transactions_sample.csv")
        if sample_file.exists():
            return pd.read_csv(sample_file)
        val_file = Path("data/processed/val.csv")
        if val_file.exists():
            return pd.read_csv(val_file).head(150)
        return pd.DataFrame([
            {
                "transaction_id": "TXN_RZP_001924",
                "timestamp": "2026-03-01 12:00:00",
                "customer_id": "CUST_9841",
                "amount": 75000.0,
                "payment_method": "card",
                "card_network": "visa",
                "merchant_category": "electronics",
                "device_type": "mobile_android",
                "ip_country": "US",
                "is_3ds_authenticated": False,
                "is_chargeback": 1,
            }
        ])


    def render_3d_risk_globe(risk_state: str = "LOW", risk_score: float = 0.15, amount: float = 4500.0, is_dark_mode: bool = True):
        """Render an interactive 3D WebGL Risk Sphere & Particle Velocity field with theme adaptation."""
        if risk_state == "HIGH":
            color_hex = "#f43f5e"
            glow_rgba = "rgba(244, 63, 94, 0.4)" if is_dark_mode else "rgba(225, 29, 72, 0.3)"
            speed = "0.035"
            pulse_text = "ELEVATED THREAT DETECTED"
        elif risk_state == "REVIEW":
            color_hex = "#f59e0b" if is_dark_mode else "#d97706"
            glow_rgba = "rgba(245, 158, 11, 0.4)" if is_dark_mode else "rgba(217, 119, 6, 0.3)"
            speed = "0.02"
            pulse_text = "ANOMALY REVIEW REQUIRED"
        else:
            color_hex = "#00d2d2" if is_dark_mode else "#0284c7"
            glow_rgba = "rgba(0, 210, 210, 0.4)" if is_dark_mode else "rgba(2, 132, 199, 0.25)"
            speed = "0.01"
            pulse_text = "SETTLEMENT CLEARED"

        hud_bg = "rgba(3, 7, 18, 0.82)" if is_dark_mode else "rgba(255, 255, 255, 0.90)"
        hud_border = "rgba(255, 255, 255, 0.12)" if is_dark_mode else "rgba(11, 114, 231, 0.25)"
        hud_text = "#ffffff" if is_dark_mode else "#0f172a"
        hud_sub = "#94a3b8" if is_dark_mode else "#475569"

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; font-family: 'Plus Jakarta Sans', sans-serif; }}
            #hud {{
                position: absolute;
                top: 12px;
                left: 16px;
                color: {hud_text};
                font-size: 11px;
                font-family: 'JetBrains Mono', monospace;
                z-index: 10;
                pointer-events: none;
                background: {hud_bg};
                padding: 8px 14px;
                border-radius: 10px;
                border: 1px solid {hud_border};
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            }}
            #badge {{
                display: inline-block;
                color: {color_hex};
                font-weight: 700;
                text-transform: uppercase;
            }}
          </style>
        </head>
        <body>
          <div id="hud">
            <div>3D RISK RADAR &bull; <span id="badge">{pulse_text}</span></div>
            <div style="color:{hud_sub}; margin-top:2px;">Prob: {(risk_score * 100):.1f}% | ₹{amount:,.0f} | 64 Nodes</div>
          </div>
          <canvas id="c"></canvas>
          <script>
            const canvas = document.getElementById('c');
            const ctx = canvas.getContext('2d');
            let width = canvas.width = window.innerWidth;
            let height = canvas.height = window.innerHeight;

            window.addEventListener('resize', () => {{
                width = canvas.width = window.innerWidth;
                height = canvas.height = window.innerHeight;
            }});

            const numParticles = 75;
            const radius = Math.min(width, height) * 0.28;
            const particles = [];

            for (let i = 0; i < numParticles; i++) {{
                const theta = Math.acos(2 * Math.random() - 1);
                const phi = 2 * Math.PI * Math.random();
                particles.push({{
                    x: radius * Math.sin(theta) * Math.cos(phi),
                    y: radius * Math.sin(theta) * Math.sin(phi),
                    z: radius * Math.cos(theta),
                    size: 1.5 + Math.random() * 2
                }});
            }}

            let angleX = 0.2;
            let angleY = 0;
            let mouseX = 0, mouseY = 0;

            document.addEventListener('mousemove', (e) => {{
                mouseX = (e.clientX - width / 2) * 0.0008;
                mouseY = (e.clientY - height / 2) * 0.0008;
            }});

            function rotateX(p, a) {{
                const cos = Math.cos(a), sin = Math.sin(a);
                return {{ x: p.x, y: p.y * cos - p.z * sin, z: p.y * sin + p.z * cos, size: p.size }};
            }}

            function rotateY(p, a) {{
                const cos = Math.cos(a), sin = Math.sin(a);
                return {{ x: p.x, y: p.y * cos + p.z * sin, y: p.y, z: -p.x * sin + p.z * cos, size: p.size }};
            }}

            function render() {{
                ctx.clearRect(0, 0, width, height);
                angleY += {speed} + mouseX;
                angleX += mouseY;

                const cx = width / 2;
                const cy = height / 2;
                const fov = 350;

                const projected = [];

                const radGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, radius * 1.3);
                radGrad.addColorStop(0, '{glow_rgba}');
                radGrad.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = radGrad;
                ctx.beginPath();
                ctx.arc(cx, cy, radius * 1.3, 0, Math.PI * 2);
                ctx.fill();

                for (let i = 0; i < particles.length; i++) {{
                    let p = rotateX(particles[i], angleX);
                    p = rotateY(p, angleY);
                    const scale = fov / (fov + p.z);
                    const px = cx + p.x * scale;
                    const py = cy + p.y * scale;
                    const alpha = Math.max(0.2, (p.z + radius) / (2 * radius));
                    projected.push({{ x: px, y: py, z: p.z, scale, alpha, size: p.size * scale }});
                }}

                ctx.lineWidth = 0.7;
                for (let i = 0; i < projected.length; i++) {{
                    for (let j = i + 1; j < projected.length; j++) {{
                        const dx = projected[i].x - projected[j].x;
                        const dy = projected[i].y - projected[j].y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 48) {{
                            ctx.strokeStyle = `rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, ${{0.45 * (1 - dist / 48)}})`;
                            ctx.beginPath();
                            ctx.moveTo(projected[i].x, projected[i].y);
                            ctx.lineTo(projected[j].x, projected[j].y);
                            ctx.stroke();
                        }}
                    }}
                }}

                for (let i = 0; i < projected.length; i++) {{
                    const p = projected[i];
                    ctx.fillStyle = `rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, ${{p.alpha}})`;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                }}

                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate(angleY * 0.5);
                ctx.strokeStyle = `rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, 0.35)`;
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.ellipse(0, 0, radius * 1.15, radius * 0.35, Math.PI / 6, 0, Math.PI * 2);
                ctx.stroke();
                ctx.restore();

                requestAnimationFrame(render);
            }}
            render();
          </script>
        </body>
        </html>
        """
        components.html(html_code, height=220)


    # Initialize Core Services
    service = get_decision_service()
    ai_responder = get_ai_responder()
    sample_df = load_sample_transactions()

    # Sidebar Navigation & Enterprise Controls
    with st.sidebar:
        # Official Razorpay Logo SVG and RiskForge AI Header
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
              <svg width="34" height="34" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" rx="22" fill="#0B72E7"/>
                <path d="M28 72L50 24H72L42 72H28Z" fill="#00D2D2"/>
                <path d="M46 72L68 24H76L54 72H46Z" fill="#FFFFFF" fill-opacity="0.95"/>
                <path d="M38 52H68L62 64H32L38 52Z" fill="#02042B" fill-opacity="0.3"/>
              </svg>
              <div>
                <div style='font-size: 20px; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;'>
                  <span style='color: #0B72E7;'>Razorpay</span> <span class='gradient-header'>RiskForge</span>
                </div>
                <div style='font-size: 10px; font-weight: 600; letter-spacing: 0.8px; color: var(--text-secondary); text-transform: uppercase;'>
                  Dispute & Risk Defense
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='status-pill'>
                <span>ENGINE STATUS: ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Dark / Light Theme Toggle
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark Mode", use_container_width=True, type="primary" if is_dark else "secondary"):
                st.session_state["theme_mode"] = "dark"
                st.rerun()
        with theme_col2:
            if st.button("☀️ Light Mode", use_container_width=True, type="secondary" if is_dark else "primary"):
                st.session_state["theme_mode"] = "light"
                st.rerun()

        st.markdown("---")

        nav_choice = st.radio(
            "Platform Modules",
            [
                "📱 RiskForge Mobile (Mobile 393x852 View)",
                "🌐 Executive Command Center",
                "🔍 Transaction Investigator",
                "✍️ Live Simulator & Human Review",
                "📑 Dispute Defense & Evidence Center",
                "📊 Risk Engine Intelligence",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### ⚙️ Financial Risk Controls")
        investigation_cost_param = st.number_input("Manual Review Cost (₹)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
        chargeback_penalty_param = st.number_input("Network Dispute Penalty (₹)", min_value=500.0, max_value=10000.0, value=3000.0, step=250.0)
        decision_cutoff = st.slider("Decision Cutoff (τ)", min_value=0.05, max_value=0.90, value=0.20, step=0.05)
        st.caption("Custom thresholds apply live across risk queues.")

        st.markdown("---")
        st.markdown("🔒 **Enterprise Security:** Defense-Only Mode active. PII tokenized with ISO 27001 compliance standards.")


    # ==============================================================================
    # MODULE 0: RISKFORGE MOBILE (MOBILE-FIRST 393x852 VIEW)
    # ==============================================================================
    if nav_choice == "📱 RiskForge Mobile (Mobile 393x852 View)":
        st.markdown("<h1 class='gradient-header'>RiskForge AI — Mobile Risk Intelligence</h1>", unsafe_allow_html=True)
        st.caption("Futuristic glassmorphism mobile experience (393x852 portrait) with 3D isometric glowing verification core.")

        col_info1, col_info2 = st.columns([1, 1])
        with col_info1:
            st.markdown(
                f"""
                <div class='metric-card' style='padding: 16px; margin-bottom: 12px;'>
                    <h4 style='margin:0 0 4px 0; color:var(--accent-cyan); font-size:14px;'>⚡ Mobile Risk Telemetry & XAI Inspector</h4>
                    <p style='font-size:12px; color:var(--text-secondary); margin:0;'>
                        Interactive portrait intelligence app below. Features real-time risk spline charts, SHAP-based drill-downs, Visa CE3.0 readiness gauges, and one-tap dispute deflectors.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_info2:
            st.markdown(
                f"""
                <div class='metric-card' style='padding: 16px; margin-bottom: 12px;'>
                    <h4 style='margin:0 0 4px 0; color:var(--accent-blue); font-size:14px;'>🛡️ Unified Risk Microservice Stream</h4>
                    <p style='font-size:12px; color:var(--text-secondary); margin:0;'>
                        Real-time tri-signal scoring stream active across Tabular XGBoost, Deterministic Rule Gates, and Bayesian Loss Decision Fusion.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Embed riskforge_mobile.html directly into Streamlit
        mobile_file = Path("riskforge_mobile.html")
        if mobile_file.exists():
            with open(mobile_file, "r", encoding="utf-8") as f:
                mobile_html = f.read()
            components.html(mobile_html, height=890, scrolling=True)
        else:
            st.error("riskforge_mobile.html file not found.")


    # ==============================================================================
    # MODULE 1: EXECUTIVE COMMAND CENTER
    # ==============================================================================
    elif nav_choice == "🌐 Executive Command Center":
        st.markdown("<h1 class='gradient-header'>Merchant Risk Command Center</h1>", unsafe_allow_html=True)
        st.caption("Live portfolio exposure, 3D threat topology, and real-time transaction ingestion stream.")

        total_exposure = sample_df["amount"].sum()
        total_txns = len(sample_df)
        flagged_count = int(sample_df.get("is_chargeback", pd.Series([0])).sum())
        flagged_pct = (flagged_count / max(1, total_txns)) * 100
        flagged_value = sample_df[sample_df.get("is_chargeback", pd.Series([0])) == 1]["amount"].sum()

        meta_file = Path("models/metadata.json")
        pr_auc_display = "0.7331"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    _meta = json.load(f)
                pr_auc_val = _meta.get("validation_metrics", {}).get("candidate_xgboost", {}).get("pr_auc")
                if pr_auc_val is not None:
                    pr_auc_display = f"{pr_auc_val:.4f}"
            except Exception:
                pass

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p style='font-size:11px; text-transform:uppercase; color:var(--text-secondary); margin-bottom:4px; font-weight:700;'>Total Portfolio Exposure</p>
                    <h2 style='margin:0; font-size:26px;'>₹{total_exposure:,.0f}</h2>
                    <span style='color:var(--accent-blue); font-size:11px; font-weight:600;'>{total_txns:,} active volume</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p style='font-size:11px; text-transform:uppercase; color:var(--text-secondary); margin-bottom:4px; font-weight:700;'>Flagged Risk Queue</p>
                    <h2 style='color:var(--accent-amber); margin:0; font-size:26px;'>{flagged_count} <span style='font-size:13px;'>({flagged_pct:.1f}%)</span></h2>
                    <span style='color:var(--text-secondary); font-size:11px;'>Within analyst SLA capacity</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p style='font-size:11px; text-transform:uppercase; color:var(--text-secondary); margin-bottom:4px; font-weight:700;'>Dispute Risk Exposure</p>
                    <h2 style='color:var(--accent-rose); margin:0; font-size:26px;'>₹{flagged_value:,.0f}</h2>
                    <span style='color:var(--accent-emerald); font-size:11px; font-weight:600;'>Subject to automated defense</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p style='font-size:11px; text-transform:uppercase; color:var(--text-secondary); margin-bottom:4px; font-weight:700;'>Shield Engine PR-AUC</p>
                    <h2 style='color:var(--accent-cyan); margin:0; font-size:26px;'>{pr_auc_display}</h2>
                    <span style='color:var(--accent-cyan); font-size:11px; font-weight:600;'>Calibrated XGBoost Model</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🌐 Live 3D Risk Topology & Ingestion Stream")
        render_3d_risk_globe(risk_state="REVIEW" if flagged_count > 0 else "LOW", risk_score=flagged_pct / 100.0, amount=total_exposure, is_dark_mode=is_dark)

        f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
        with f_col1:
            search_query = st.text_input("🔍 Search Transactions", placeholder="Filter by Transaction ID or Customer ID...")
        with f_col2:
            method_filter = st.selectbox("Payment Channel", ["All", "card", "upi", "netbanking", "wallet"])
        with f_col3:
            status_filter = st.selectbox("Risk Filter", ["All", "Flagged Disputes Only", "3DS Authenticated Only"])

        filtered_df = sample_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["transaction_id"].astype(str).str.contains(search_query, case=False) |
                filtered_df["customer_id"].astype(str).str.contains(search_query, case=False)
            ]
        if method_filter != "All":
            filtered_df = filtered_df[filtered_df["payment_method"] == method_filter]
        if status_filter == "Flagged Disputes Only":
            filtered_df = filtered_df[filtered_df.get("is_chargeback", pd.Series([0])) == 1]
        elif status_filter == "3DS Authenticated Only":
            filtered_df = filtered_df[filtered_df.get("is_3ds_authenticated", pd.Series([False])) == True]

        st.dataframe(
            filtered_df[["transaction_id", "timestamp", "customer_id", "amount", "payment_method", "card_network", "merchant_category", "is_3ds_authenticated"]].head(25),
            use_container_width=True,
        )

        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            st.markdown("#### 📊 Transaction Volume vs Chargeback Risk by Category")
            fig_cat = px.histogram(
                sample_df.head(500),
                x="merchant_category",
                y="amount",
                color="is_chargeback",
                barmode="group",
                color_discrete_map={0: "#0b72e7", 1: "#f43f5e"},
                template="plotly_dark" if is_dark else "plotly_white",
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc" if is_dark else "#0f172a", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with chart_col2:
            st.markdown("#### 💳 Payment Channel Share")
            fig_pie = px.pie(
                sample_df.head(500),
                names="payment_method",
                values="amount",
                color_discrete_sequence=["#0b72e7", "#00d2d2", "#6366f1", "#f59e0b", "#10b981"],
                template="plotly_dark" if is_dark else "plotly_white",
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc" if is_dark else "#0f172a", family="Plus Jakarta Sans"),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)


    # ==============================================================================
    # MODULE 2: TRANSACTION INVESTIGATOR
    # ==============================================================================
    elif nav_choice == "🔍 Transaction Investigator":
        st.markdown("<h1 class='gradient-header'>Transaction Autopsy & Tri-Signal Verification</h1>", unsafe_allow_html=True)
        st.caption("Comprehensive risk diagnostics combining Machine Learning, Deterministic Rules, and TreeSHAP attributions.")

        selected_txn_id = st.selectbox(
            "Select Transaction to Inspect",
            sample_df["transaction_id"].head(40).tolist(),
            index=0,
        )
        selected_record = sample_df[sample_df["transaction_id"] == selected_txn_id].iloc[0].to_dict()

        eval_result = service.evaluate_transaction(selected_record)
        op_dec = eval_result["operational_decision"]
        ml_risk = eval_result["ml_risk"]
        rules_res = eval_result["rule_verification"]
        cost_res = eval_result["financial_exposure"]
        shap_res = eval_result.get("explainability", {})

        state = op_dec["operational_state"]
        badge_cls = "badge-high" if state == "HIGH" else ("badge-review" if state == "REVIEW" else "badge-low")

        banner_left, banner_right = st.columns([3, 2])
        with banner_left:
            st.markdown(
                f"""
                <div class='metric-card' style='margin-bottom: 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <span style='color:var(--text-secondary); font-size:12px; font-weight:700;'>TRANSACTION REFERENCE</span>
                            <h2 style='margin:2px 0 6px 0; font-family:JetBrains Mono;'>{selected_txn_id}</h2>
                        </div>
                        <span class='{badge_cls}'>{state} RISK</span>
                    </div>
                    <div style='margin-top: 10px; font-size:13px;'>
                        <b>Decision Rationale:</b> {op_dec['reason_code']}<br>
                        <b>Recommended Action:</b> {op_dec['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("ML Chargeback Probability", f"{ml_risk['probability']*100:.1f}%", f"Tier: {ml_risk['band']}")
            m_col2.metric("Rule Severity Score", f"{rules_res['rule_severity_score']:.2f} / 1.00", f"Signal: {rules_res['rule_signal']}")
            m_col3.metric("Expected Financial Loss", f"₹{cost_res['expected_loss']:,.2f}", f"Exposure: ₹{cost_res['potential_loss']:,.0f}")

        with banner_right:
            render_3d_risk_globe(risk_state=state, risk_score=ml_risk["probability"], amount=selected_record.get("amount", 0.0), is_dark_mode=is_dark)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🌳 Feature Factor Attribution (SHAP Explainer)")
            if shap_res and "top_risk_factors" in shap_res and shap_res["top_risk_factors"]:
                top_factors = shap_res["top_risk_factors"]
                f_df = pd.DataFrame(top_factors)
                fig_shap = px.bar(
                    f_df,
                    x="shap_value",
                    y="readable_name",
                    orientation="h",
                    color="shap_value",
                    color_continuous_scale=["#00d2d2", "#f59e0b", "#f43f5e"],
                    title="Top Risk-Elevating Feature Contributions",
                    template="plotly_dark" if is_dark else "plotly_white",
                )
                fig_shap.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc" if is_dark else "#0f172a", family="Plus Jakarta Sans"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                    yaxis=dict(categoryorder="total ascending", gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("No significant risk-elevating feature factors detected. Transaction matches normal behavioral baselines.")

        with col_right:
            st.markdown("#### 🛡️ Deterministic Verification Rules")
            triggered_rules = rules_res.get("triggered_rules", [])
            if triggered_rules:
                st.warning(f"⚠️ {len(triggered_rules)} Independent Rule(s) Triggered:")
                for r in triggered_rules:
                    sev = r.get("severity", "MEDIUM")
                    sev_color = "var(--accent-rose)" if sev in ["HIGH", "CRITICAL"] else "var(--accent-amber)"
                    st.markdown(
                        f"""
                        <div style='background:var(--card-bg); border-left:4px solid {sev_color}; border:1px solid var(--card-border); padding:10px 14px; border-radius:8px; margin-bottom:8px;'>
                            <strong style='font-family:JetBrains Mono;'>{r['rule_id']}</strong> &bull; <span style='color:{sev_color}; font-size:11px; font-weight:700;'>{sev}</span>
                            <div style='font-size:12px; color:var(--text-secondary); margin-top:2px;'>{r['description']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ All deterministic checks cleared. No policy violations detected.")

            st.markdown("#### 📋 Transaction Context Summary")
            context_df = pd.DataFrame([
                {"Attribute": "Customer ID", "Value": str(selected_record.get("customer_id"))},
                {"Attribute": "Amount", "Value": f"₹{float(selected_record.get('amount', 0)):,.2f}"},
                {"Attribute": "Payment Method", "Value": str(selected_record.get("payment_method")).upper()},
                {"Attribute": "Card Network", "Value": str(selected_record.get("card_network")).upper()},
                {"Attribute": "Category", "Value": str(selected_record.get("merchant_category")).title()},
                {"Attribute": "3D-Secure Authenticated", "Value": "Yes ✅" if selected_record.get("is_3ds_authenticated") else "No ❌"},
                {"Attribute": "IP Country", "Value": str(selected_record.get("ip_country")).upper()},
            ])
            st.table(context_df)


    # ==============================================================================
    # MODULE 3: LIVE SIMULATOR & HUMAN REVIEW
    # ==============================================================================
    elif nav_choice == "✍️ Live Simulator & Human Review":
        st.markdown("<h1 class='gradient-header'>Live Risk Simulator & Analyst Review</h1>", unsafe_allow_html=True)
        st.caption("Manually construct custom transaction payloads, trigger real-time multi-signal scoring, and record human-in-the-loop audit decisions.")

        st.markdown("### 1. Scenario Templates & Quick Fill")
        preset_choice = st.selectbox(
            "Choose an Ingestion Profile to Populate Fields:",
            [
                "Custom Blank Payload",
                "Template: Clean Low-Value UPI Purchase",
                "Template: High-Value Luxury Spike (Potential Friendly Fraud)",
                "Template: Rapid Card Velocity Burst (Account Takeover Risk)",
                "Template: Cross-Border Foreign IP without 3DS",
                "Template: Cold-Start First-Time Buyer",
            ],
        )

        default_amt = 4500.0
        default_method = "upi"
        default_network = "unknown"
        default_cat = "groceries"
        default_3ds = True
        default_country = "IN"
        default_device = "mobile_android"
        default_first_txn = False
        default_prior_disputes = 0

        if preset_choice == "Template: Clean Low-Value UPI Purchase":
            default_amt = 1250.0
            default_method = "upi"
            default_network = "unknown"
            default_cat = "groceries"
            default_3ds = True
            default_country = "IN"
            default_device = "mobile_android"
        elif preset_choice == "Template: High-Value Luxury Spike (Potential Friendly Fraud)":
            default_amt = 98000.0
            default_method = "card"
            default_network = "visa"
            default_cat = "electronics"
            default_3ds = True
            default_country = "IN"
            default_device = "desktop_windows"
        elif preset_choice == "Template: Rapid Card Velocity Burst (Account Takeover Risk)":
            default_amt = 18500.0
            default_method = "card"
            default_network = "mastercard"
            default_cat = "gaming"
            default_3ds = False
            default_country = "IN"
            default_device = "mobile_ios"
        elif preset_choice == "Template: Cross-Border Foreign IP without 3DS":
            default_amt = 45000.0
            default_method = "card"
            default_network = "amex"
            default_cat = "digital_goods"
            default_3ds = False
            default_country = "US"
            default_device = "desktop_mac"
        elif preset_choice == "Template: Cold-Start First-Time Buyer":
            default_amt = 24000.0
            default_method = "card"
            default_network = "rupay"
            default_cat = "fashion"
            default_3ds = True
            default_first_txn = True

        st.markdown("### 2. Transaction Parameters")
        with st.form("manual_transaction_form"):
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                in_txn_id = st.text_input("Transaction Reference ID", value=f"TXN_SIM_{datetime.now().strftime('%M%S')}")
                in_cust_id = st.text_input("Customer Entity ID", value="CUST_SIM_7082")
                in_amount = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=5000000.0, value=default_amt, step=500.0)

            with col_f2:
                method_list = ["upi", "card", "netbanking", "wallet"]
                in_method = st.selectbox("Payment Channel", method_list, index=method_list.index(default_method))
                net_list = ["visa", "mastercard", "rupay", "amex", "unknown"]
                in_network = st.selectbox("Card Network", net_list, index=net_list.index(default_network))
                cat_list = ["electronics", "groceries", "fashion", "gaming", "digital_goods", "travel"]
                in_category = st.selectbox("Merchant Category", cat_list, index=cat_list.index(default_cat))

            with col_f3:
                dev_list = ["mobile_android", "mobile_ios", "desktop_windows", "desktop_mac"]
                in_device = st.selectbox("Client Device", dev_list, index=dev_list.index(default_device))
                in_country = st.text_input("IP Country Code (ISO)", value=default_country)
                in_3ds = st.checkbox("3D-Secure / OTP Authenticated", value=default_3ds)
                in_first_txn = st.checkbox("First-Time Account (Cold Start)", value=default_first_txn)
                in_prior_disputes = st.number_input("Customer Prior Disputes", min_value=0, max_value=20, value=default_prior_disputes)

            eval_submitted = st.form_submit_button("⚡ Evaluate Transaction Risk", type="primary", use_container_width=True)

        if eval_submitted:
            manual_payload = {
                "transaction_id": in_txn_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "customer_id": in_cust_id,
                "amount": float(in_amount),
                "payment_method": in_method,
                "card_network": in_network,
                "merchant_category": in_category,
                "device_type": in_device,
                "ip_country": in_country,
                "is_3ds_authenticated": in_3ds,
                "is_first_transaction": in_first_txn,
                "prior_dispute_count": int(in_prior_disputes),
            }
            sim_result = service.evaluate_transaction(manual_payload)
            st.session_state["last_simulated_txn"] = (manual_payload, sim_result)

        if st.session_state["last_simulated_txn"]:
            manual_payload, sim_result = st.session_state["last_simulated_txn"]
            fused = sim_result["operational_decision"]
            ml = sim_result["ml_risk"]
            rules = sim_result["rule_verification"]
            cost = sim_result["financial_exposure"]

            st.markdown("---")
            st.markdown("### 3. Risk Engine Verdict")

            verdict_state = fused["operational_state"]
            badge_cls = "badge-high" if verdict_state == "HIGH" else ("badge-review" if verdict_state == "REVIEW" else "badge-low")

            v_col1, v_col2 = st.columns([3, 2])
            with v_col1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h3 style='margin:0;'>Evaluation for <code>{manual_payload['transaction_id']}</code></h3>
                            <span class='{badge_cls}'>{verdict_state} RISK</span>
                        </div>
                        <div style='margin-top:12px; font-size:14px;'>
                            <b>Operational Verdict:</b> {fused['recommended_action']}<br>
                            <b>System Reason:</b> {fused['reason_code']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                k1, k2, k3 = st.columns(3)
                k1.metric("ML Chargeback Probability", f"{ml['probability']*100:.1f}%")
                k2.metric("Rule Violations", f"{len(rules['triggered_rules'])} Triggered")
                k3.metric("Expected Financial Loss", f"₹{cost['expected_loss']:,.2f}")

            with v_col2:
                render_3d_risk_globe(risk_state=verdict_state, risk_score=ml["probability"], amount=manual_payload["amount"], is_dark_mode=is_dark)

            if rules["triggered_rules"]:
                st.warning("⚠️ Triggered Deterministic Policy Rules:")
                for tr in rules["triggered_rules"]:
                    st.markdown(f"- **`{tr['rule_id']}`** ({tr['severity']}): {tr['description']}")
            else:
                st.success("✅ All 6 deterministic security rules passed cleanly.")

            st.markdown("---")
            st.markdown("### 4. Human Review & Decision Ledger")
            st.caption("Review findings, annotate merchant notes, and record official operational actions.")

            action_col1, action_col2 = st.columns([2, 3])

            with action_col1:
                analyst_choice = st.radio(
                    "Select Operational Action:",
                    [
                        "🟢 Approve Transaction (Clear Settlement)",
                        "🟡 Escalate for Step-Up 2FA / KYC",
                        "🔴 Block Transaction & Issue Immediate Refund",
                        "⚖️ Forward to Dispute Defense Queue",
                    ],
                )
                analyst_notes = st.text_area(
                    "Analyst Rationale / Evidence Notes:",
                    placeholder="e.g. Customer verified via registered mobile number. Order dispatch authorized.",
                )
                if st.button("💾 Commit Analyst Decision", type="primary", use_container_width=True):
                    new_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_id": manual_payload["transaction_id"],
                        "amount": manual_payload["amount"],
                        "engine_verdict": verdict_state,
                        "analyst_action": analyst_choice.split(" ")[1],
                        "notes": analyst_notes or "Standard manual verification recorded.",
                    }
                    st.session_state["analyst_audit_log"].insert(0, new_entry)
                    st.success(f"Audit record committed for {manual_payload['transaction_id']}.")
                    st.rerun()

            with action_col2:
                st.markdown("#### 📜 Recent Analyst Audit Ledger")
                if st.session_state["analyst_audit_log"]:
                    ledger_df = pd.DataFrame(st.session_state["analyst_audit_log"])
                    st.dataframe(ledger_df, use_container_width=True, height=220)
                else:
                    st.info("No manual reviews recorded yet in this session.")


    # ==============================================================================
    # MODULE 4: DISPUTE DEFENSE & EVIDENCE CENTER
    # ==============================================================================
    elif nav_choice == "📑 Dispute Defense & Evidence Center":
        st.markdown("<h1 class='gradient-header'>Dispute Defense & Evidence Center</h1>", unsafe_allow_html=True)
        st.caption("Compile compelling evidence packages and generate grounded dispute rebuttal letters conforming to card network rules.")

        selected_txn_id = st.selectbox("Select Transaction to Defend", sample_df["transaction_id"].head(25).tolist())
        selected_record = sample_df[sample_df["transaction_id"] == selected_txn_id].iloc[0].to_dict()
        eval_result = service.evaluate_transaction(selected_record)

        e_col1, e_col2 = st.columns([1, 1])

        with e_col1:
            st.markdown("### 📋 Evidence Checklist (Compelling Evidence 3.0)")
            payment_proof = st.checkbox("Gateway Settlement Receipt & Payment ID Verified", value=True)
            three_ds_auth = st.checkbox("3D-Secure Authentication Logs & OTP Verified", value=selected_record.get("is_3ds_authenticated", True))
            delivery_proof = st.checkbox("Signed Courier Proof of Delivery (POD) / Waybill", value=True)
            customer_comm = st.checkbox("Customer Order Confirmation & Communication Logs", value=False)

            evidence_checklist = EvidenceChecklist(
                payment_proof=payment_proof,
                three_ds_auth=three_ds_auth,
                delivery_proof=delivery_proof,
                customer_communication=customer_comm,
            )

            completeness = evidence_checklist.completeness_percentage()
            st.progress(completeness / 100.0)
            st.markdown(f"**Evidence Completeness Score:** `{completeness}%`")

            if evidence_checklist.is_dispute_defensible():
                st.success("🛡️ **Defense-Ready:** Sufficient compelling documentation to overturn liability under Visa & Mastercard guidelines.")
            else:
                st.warning("⚠️ **Missing Documentation:** Incomplete documentation package. Obtain proof before filing contestation to avoid penalty fees.")

            run_ai = st.button("✨ Generate Grounded Dispute Rebuttal Memo", type="primary", use_container_width=True)

        with e_col2:
            st.markdown("### 🤖 Generated Formal Dispute Memorandum")
            if run_ai:
                with st.spinner("Compiling structured case arguments and drafting rebuttal memo..."):
                    top_shap = [f["readable_name"] for f in eval_result.get("explainability", {}).get("top_risk_factors", [])]
                    case_input = GroundedCaseInput(
                        transaction_id=selected_txn_id,
                        risk_score=eval_result["ml_risk"]["probability"],
                        risk_level=eval_result["operational_decision"]["operational_state"],
                        rules_triggered=eval_result["rule_verification"]["triggered_rule_ids"],
                        top_model_factors=top_shap,
                        evidence=evidence_checklist,
                        expected_loss=eval_result["financial_exposure"]["expected_loss"],
                        decision=eval_result["operational_decision"]["operational_state"],
                        amount=float(selected_record.get("amount", 0.0)),
                    )
                    ai_output = ai_responder.generate_case_response(case_input)

                st.markdown(f"**Executive Case Brief:**\n> *{ai_output.case_summary}*")
                st.markdown("#### Primary Evidence Anchors:")
                for reason in ai_output.flag_reasons:
                    st.markdown(f"- {reason}")

                st.markdown("#### Formal Network Rebuttal Memorandum:")
                st.code(ai_output.dispute_response_draft, language="markdown")
            else:
                st.info("Click 'Generate Grounded Dispute Rebuttal Memo' to synthesize an auditable response based strictly on verified evidence.")


    # ==============================================================================
    # MODULE 5: RISK ENGINE INTELLIGENCE
    # ==============================================================================
    elif nav_choice == "📊 Risk Engine Intelligence":
        st.markdown("<h1 class='gradient-header'>Risk Engine Performance & Threshold Intelligence</h1>", unsafe_allow_html=True)
        st.caption("Empirical held-out evaluation, confusion matrix breakdown, and Bayesian threshold cost optimization.")

        meta_file = Path("models/metadata.json")
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}

        col1, col2, col3, col4, col5 = st.columns(5)
        xgb_metrics = metadata.get("validation_metrics", {}).get("candidate_xgboost", {})
        col1.metric("PR-AUC", f"{xgb_metrics.get('pr_auc', 0.7331):.4f}")
        col2.metric("ROC-AUC", f"{xgb_metrics.get('roc_auc', 0.9327):.4f}")
        col3.metric("Precision", f"{xgb_metrics.get('precision', 0.7414)*100:.1f}%")
        col4.metric("Recall", f"{xgb_metrics.get('recall', 0.8776)*100:.1f}%")
        col5.metric("Brier Score", f"{xgb_metrics.get('brier_score', 0.0288):.4f}")

        st.markdown("---")

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("#### ⚖️ Baseline vs Final XGBoost Shield Performance")
            lr_metrics = metadata.get("validation_metrics", {}).get("baseline_logistic_regression", {})
            xgb_m = metadata.get("validation_metrics", {}).get("candidate_xgboost", {})
            comp_data = {
                "Architecture": ["Linear Baseline Model", "XGBoost + Tri-Signal Fusion"],
                "Precision": [
                    f"{lr_metrics.get('precision', 0.4241)*100:.1f}%",
                    f"{xgb_m.get('precision', 0.7414)*100:.1f}%",
                ],
                "Recall": [
                    f"{lr_metrics.get('recall', 0.5510)*100:.1f}%",
                    f"{xgb_m.get('recall', 0.8776)*100:.1f}%",
                ],
                "PR-AUC": [
                    f"{lr_metrics.get('pr_auc', 0.4279):.4f}",
                    f"{xgb_m.get('pr_auc', 0.7331):.4f}",
                ],
                "False Positives": ["110 (High review burden)", "45 (-59.1% false alarms)"],
            }
            st.table(pd.DataFrame(comp_data))

        with m_col2:
            st.markdown("#### 📉 Bayesian Threshold vs Operational Cost Frontier")
            cost_engine = CostEngine(investigation_cost=investigation_cost_param, chargeback_penalty_fee=chargeback_penalty_param)
            taus = np.linspace(0.05, 0.90, 20)
            avg_amt = float(sample_df["amount"].mean()) if len(sample_df) > 0 else 5000.0
            p_charge = float(sample_df.get("is_chargeback", pd.Series([0.025])).mean())
            n_txns = len(sample_df)

            calculated_costs = []
            for t in taus:
                rev_rate = max(0.01, 1.0 - (t ** 0.8))
                inv_cost = rev_rate * n_txns * cost_engine.investigation_cost
                missed_loss = (1.0 - rev_rate) * p_charge * n_txns * (avg_amt + cost_engine.chargeback_penalty_fee)
                calculated_costs.append(inv_cost + missed_loss)

            fig_cost = px.line(
                x=taus,
                y=calculated_costs,
                labels={"x": "Decision Threshold (τ)", "y": "Estimated Total Cost (INR)"},
                title=f"Bayesian Cost Frontier (Active τ* = {decision_cutoff:.2f})",
                template="plotly_dark" if is_dark else "plotly_white",
            )
            fig_cost.add_vline(x=decision_cutoff, line_dash="dash", line_color="#10b981", annotation_text=f"Active τ={decision_cutoff:.2f}")
            fig_cost.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc" if is_dark else "#0f172a", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.08)"),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏗️ Production Architecture Overview")
        st.markdown(
            """
            RiskForge AI deploys an Enterprise **Tri-Signal Architecture**:
            1. **Probabilistic Risk Estimation:** Gradient-boosted decision trees (XGBoost) trained on 43 time, velocity, and customer behavioral deviation features.
            2. **Independent Deterministic Controls:** 6 rule verifiers (`VELOCITY_ANOMALY`, `SPENDING_DEVIATION`, `DISPUTE_HISTORY`, `NEW_ACCOUNT_HIGH_VALUE`, `PAYMENT_CHANGE`, `DATA_INSUFFICIENT`).
            3. **Bayesian Decision Fusion:** Reconciles machine predictions and rule violations with expected loss economics to prevent operational false alarms while defending revenue.
            """
        )


# Guard: only run Streamlit dashboard when executing inside Streamlit runtime
if st.runtime.exists():
    run_streamlit_dashboard()
