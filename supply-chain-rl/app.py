# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from supply_chain_env import SupplyChainEnv, Action
from stable_baselines3 import A2C, PPO

EVAL_SEEDS = [11, 22, 33, 44, 55]

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Light Mode Theme Data */
:root,
.gradio-container.light,
:root .light {
    --bg-top: #f8fafc;
    --bg-mid: #f1f5f9;
    --bg-bottom: #e2e8f0;
    --panel: rgba(255, 255, 255, 0.85);
    --panel-strong: rgba(255, 255, 255, 0.95);
    --panel-soft: rgba(248, 250, 252, 0.7);
    --panel-border: rgba(0, 0, 0, 0.08);
    --panel-glow: rgba(168, 85, 247, 0.15);
    --ink: #0f172a;
    --muted: #64748b;
    --accent: #8b5cf6;
    --accent-2: #0d9488;
    --accent-3: #d97706;
    --accent-4: #db2777;
    --warn: #e11d48;
    --button-a: #8b5cf6;
    --button-b: #a78bfa;
    --shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
    --shadow-soft: 0 12px 40px rgba(0, 0, 0, 0.05);

    /* Gradio overrides for light mode */
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    --block-background-fill: transparent !important;
    --block-border-color: rgba(0, 0, 0, 0.08) !important;
    --block-label-background-fill: rgba(255, 255, 255, 0.8) !important;
    --block-label-border-color: rgba(0, 0, 0, 0.08) !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #64748b !important;
    --border-color-accent: rgba(139, 92, 246, 0.3) !important;
    --border-color-primary: rgba(0, 0, 0, 0.08) !important;
    --button-primary-background-fill: linear-gradient(135deg, #8b5cf6, #a78bfa) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    --button-primary-border-color: transparent !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: rgba(0, 0, 0, 0.03) !important;
    --button-secondary-background-fill-hover: rgba(139, 92, 246, 0.08) !important;
    --button-secondary-border-color: rgba(0, 0, 0, 0.1) !important;
    --button-secondary-text-color: #0f172a !important;
    --checkbox-background-color: rgba(255, 255, 255, 0.8) !important;
    --checkbox-border-color: rgba(0, 0, 0, 0.15) !important;
    --checkbox-label-background-fill: transparent !important;
    --color-accent: #8b5cf6 !important;
    --color-accent-soft: rgba(139, 92, 246, 0.15) !important;
    --input-background-fill: rgba(255, 255, 255, 0.72) !important;
    --input-background-fill-focus: rgba(255, 255, 255, 0.9) !important;
    --input-border-color: rgba(0, 0, 0, 0.15) !important;
    --input-placeholder-color: #94a3b8 !important;
    --loader-color: #8b5cf6 !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #64748b !important;
    --neutral-600: #475569 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    --panel-background-fill: rgba(255, 255, 255, 0.72) !important;
    --panel-border-color: rgba(0, 0, 0, 0.08) !important;
    --shadow-drop: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
    --shadow-drop-lg: 0 8px 28px rgba(0, 0, 0, 0.12) !important;
    --slider-color: #8b5cf6 !important;
    --table-border-color: rgba(0, 0, 0, 0.08) !important;
    --table-even-background-fill: rgba(255, 255, 255, 0.4) !important;
    --table-odd-background-fill: rgba(248, 250, 252, 0.4) !important;
    --table-row-focus: rgba(139, 92, 246, 0.1) !important;
}

/* Dark Mode Theme Data */
.gradio-container.dark,
:root.dark,
:root .dark {
    --bg-top: #08080f;
    --bg-mid: #0e0e1a;
    --bg-bottom: #151520;
    --panel: rgba(20, 20, 35, 0.72);
    --panel-strong: rgba(14, 14, 26, 0.88);
    --panel-soft: rgba(30, 30, 55, 0.6);
    --panel-border: rgba(255, 255, 255, 0.07);
    --panel-glow: rgba(168, 85, 247, 0.22);
    --ink: #f1f5f9;
    --muted: #94a3b8;
    --accent: #a78bfa;
    --accent-2: #2dd4bf;
    --accent-3: #fbbf24;
    --accent-4: #f472b6;
    --warn: #fb7185;
    --button-a: #7c3aed;
    --button-b: #a78bfa;
    --shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
    --shadow-soft: 0 18px 50px rgba(0, 0, 0, 0.18);

    --body-background-fill: #08080f !important;
    --background-fill-primary: #0e0e1a !important;
    --background-fill-secondary: #151520 !important;
    --block-background-fill: transparent !important;
    --block-border-color: rgba(255, 255, 255, 0.07) !important;
    --block-label-background-fill: rgba(20, 20, 35, 0.8) !important;
    --block-label-border-color: rgba(255, 255, 255, 0.07) !important;
    --block-label-text-color: #f1f5f9 !important;
    --block-title-text-color: #f1f5f9 !important;
    --body-text-color: #f1f5f9 !important;
    --body-text-color-subdued: #94a3b8 !important;
    --border-color-accent: rgba(167, 139, 250, 0.3) !important;
    --border-color-primary: rgba(255, 255, 255, 0.07) !important;
    --button-primary-background-fill: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #6d28d9, #8b5cf6) !important;
    --button-primary-border-color: transparent !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: rgba(255, 255, 255, 0.06) !important;
    --button-secondary-background-fill-hover: rgba(167, 139, 250, 0.12) !important;
    --button-secondary-border-color: rgba(255, 255, 255, 0.1) !important;
    --button-secondary-text-color: #f1f5f9 !important;
    --checkbox-background-color: rgba(14, 14, 24, 0.8) !important;
    --checkbox-border-color: rgba(255, 255, 255, 0.15) !important;
    --checkbox-label-background-fill: transparent !important;
    --color-accent: #a78bfa !important;
    --color-accent-soft: rgba(167, 139, 250, 0.15) !important;
    --input-background-fill: rgba(14, 14, 24, 0.72) !important;
    --input-background-fill-focus: rgba(30, 30, 45, 0.9) !important;
    --input-border-color: rgba(255, 255, 255, 0.1) !important;
    --input-placeholder-color: #64748b !important;
    --loader-color: #a78bfa !important;
    --neutral-50: #0e0e1a !important;
    --neutral-100: #151520 !important;
    --neutral-200: #1e1e30 !important;
    --neutral-300: #2a2a44 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #94a3b8 !important;
    --neutral-600: #cbd5e1 !important;
    --neutral-700: #e2e8f0 !important;
    --neutral-800: #f1f5f9 !important;
    --neutral-900: #f8fafc !important;
    --neutral-950: #ffffff !important;
    --panel-background-fill: rgba(20, 20, 35, 0.72) !important;
    --panel-border-color: rgba(255, 255, 255, 0.07) !important;
    --shadow-drop: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
    --shadow-drop-lg: 0 8px 28px rgba(0, 0, 0, 0.5) !important;
    --slider-color: #a78bfa !important;
    --table-border-color: rgba(255, 255, 255, 0.07) !important;
    --table-even-background-fill: rgba(14, 14, 24, 0.4) !important;
    --table-odd-background-fill: rgba(20, 20, 35, 0.4) !important;
    --table-row-focus: rgba(167, 139, 250, 0.1) !important;
}

.gradio-container {
    position: relative;
    overflow-x: hidden;
    background:
        radial-gradient(circle at 12% 10%, rgba(167, 139, 250, 0.18), transparent 30%),
        radial-gradient(circle at 88% 12%, rgba(45, 212, 191, 0.12), transparent 18%),
        radial-gradient(circle at 50% 100%, rgba(251, 191, 36, 0.10), transparent 20%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 38%, var(--bg-bottom) 100%);
    color: var(--ink);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.gradio-container::before,
.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
}

.gradio-container::before {
    background:
        linear-gradient(90deg, rgba(255, 255, 255, 0.045) 1px, transparent 1px),
        linear-gradient(rgba(255, 255, 255, 0.045) 1px, transparent 1px);
    background-size: 120px 120px;
    mask-image: radial-gradient(circle at center, black 10%, transparent 72%);
    opacity: 0.12;
}

.gradio-container::after {
    background:
        radial-gradient(circle at 18% 30%, rgba(167, 139, 250, 0.22), transparent 16%),
        radial-gradient(circle at 78% 62%, rgba(45, 212, 191, 0.16), transparent 14%);
    filter: blur(24px);
    opacity: 0.7;
    animation: ambient-drift 18s ease-in-out infinite alternate;
}

.app-shell {
    position: relative;
    z-index: 1;
    max-width: 1180px;
    margin: 0 auto;
}

.hero-panel,
.control-card,
.insight-card,
.metric-card,
.plot-shell,
.summary-shell,
.signal-grid-shell {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 24px;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(28px) saturate(160%);
    animation: rise-in 0.8s ease backwards;
    transition:
        transform 0.28s ease,
        box-shadow 0.28s ease,
        border-color 0.28s ease,
        background 0.28s ease;
    will-change: transform, box-shadow;
}

.hero-panel {
    position: relative;
    overflow: hidden;
    padding: 36px 36px 30px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
    animation-delay: 0.02s;
}

.hero-panel::before,
.hero-panel::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    pointer-events: none;
}

.hero-panel::before {
    width: 280px;
    height: 280px;
    right: -60px;
    top: -90px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.28) 0%, rgba(167, 139, 250, 0.02) 72%);
    filter: blur(6px);
    animation: orb-drift-a 14s ease-in-out infinite alternate;
}

.hero-panel::after {
    width: 240px;
    height: 240px;
    left: -90px;
    bottom: -130px;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.18) 0%, rgba(45, 212, 191, 0.01) 74%);
    animation: orb-drift-b 16s ease-in-out infinite alternate;
}

.hero-panel:hover,
.control-card:hover,
.insight-card:hover,
.plot-shell:hover,
.summary-shell:hover,
.signal-grid-shell:hover {
    transform: translateY(-6px);
    border-color: rgba(118, 194, 255, 0.34);
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.28), 0 0 0 1px rgba(167, 139, 250, 0.06);
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
    gap: 20px;
    align-items: center;
}

.hero-copy-block {
    max-width: 720px;
}

.hero-brow {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(167, 139, 250, 0.12);
    color: var(--accent);
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 700;
    border: 1px solid rgba(167, 139, 250, 0.18);
}

.hero-title {
    margin: 18px 0 12px;
    font-size: clamp(38px, 4vw, 64px);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: var(--ink);
}

.hero-accent {
    color: transparent;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    background-clip: text;
}

.hero-copy {
    margin: 0;
    max-width: 760px;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.7;
}

.hero-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.hero-chip {
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--ink);
    border: 1px solid rgba(255, 255, 255, 0.09);
    font-size: 13px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    transition:
        transform 0.22s ease,
        border-color 0.22s ease,
        background 0.22s ease,
        box-shadow 0.22s ease;
    animation: chip-float 6s ease-in-out infinite;
}

.hero-chip:nth-child(2) {
    animation-delay: 0.8s;
}

.hero-chip:nth-child(3) {
    animation-delay: 1.6s;
}

.hero-chip:hover {
    animation: none;
    transform: translateY(-4px);
    border-color: rgba(167, 139, 250, 0.28);
    background: rgba(167, 139, 250, 0.10);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.hero-statboard {
    display: grid;
    gap: 14px;
}

.hero-stat {
    position: relative;
    overflow: hidden;
    padding: 18px 18px 16px;
    border-radius: 20px;
    background: var(--panel-strong);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.hero-stat::after {
    content: "";
    position: absolute;
    inset: auto -30px -40px auto;
    width: 120px;
    height: 120px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.16), transparent 70%);
    pointer-events: none;
    transition: transform 0.3s ease, opacity 0.3s ease;
}

.hero-stat:hover {
    transform: translateY(-4px);
    border-color: rgba(167, 139, 250, 0.24);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 18px 34px rgba(0, 0, 0, 0.18);
}

.hero-stat:hover::after {
    transform: scale(1.18);
    opacity: 0.92;
}

.hero-stat-label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

.hero-stat-value {
    display: block;
    color: var(--ink);
    font-size: 26px;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-stat-copy {
    display: block;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.hero-side-stack {
    display: grid;
    gap: 14px;
}

.hero-visual-shell {
    position: relative;
    overflow: hidden;
    padding: 18px;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background:
        radial-gradient(circle at 12% 20%, rgba(167, 139, 250, 0.16), transparent 28%),
        radial-gradient(circle at 88% 78%, rgba(244, 114, 182, 0.14), transparent 26%),
        var(--panel-strong);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.hero-visual-shell::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.06), transparent);
    transform: translateX(-100%);
    animation: hero-sheen 6s ease-in-out infinite;
    pointer-events: none;
}

.hero-visual-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.hero-visual-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

.hero-visual-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    color: var(--ink);
    font-size: 12px;
    border: 1px solid rgba(167, 139, 250, 0.14);
    background: rgba(167, 139, 250, 0.08);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-2);
    box-shadow: 0 0 0 6px rgba(45, 212, 191, 0.12), 0 0 16px rgba(45, 212, 191, 0.28);
    animation: beacon-pulse 2s ease-in-out infinite;
}

.status-dot.alt {
    background: var(--accent);
    box-shadow: 0 0 0 6px rgba(167, 139, 250, 0.12), 0 0 16px rgba(167, 139, 250, 0.28);
}

.status-dot.warn {
    background: var(--accent-3);
    box-shadow: 0 0 0 6px rgba(251, 191, 36, 0.12), 0 0 16px rgba(251, 191, 36, 0.28);
}

.hero-radar {
    position: relative;
    min-height: 212px;
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid rgba(167, 139, 250, 0.12);
    background:
        radial-gradient(circle at center, rgba(167, 139, 250, 0.18) 0%, transparent 68%),
        var(--panel-strong);
}

.hero-radar::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
    background-size: 44px 44px;
    opacity: 0.16;
}

.hero-radar::after {
    content: "";
    position: absolute;
    inset: 10% 13%;
    border-radius: 50%;
    border: 1px dashed rgba(45, 212, 191, 0.18);
    opacity: 0.7;
}

.hero-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(167, 139, 250, 0.18);
}

.hero-ring.a {
    inset: 20% 24%;
    animation: radar-pulse 5.5s ease-in-out infinite;
}

.hero-ring.b {
    inset: 31% 34%;
    animation: radar-pulse 5.5s ease-in-out infinite 0.8s;
}

.hero-ring.c {
    inset: 42% 44%;
    animation: radar-pulse 5.5s ease-in-out infinite 1.6s;
}

.hero-sweep {
    position: absolute;
    inset: -26%;
    background: conic-gradient(
        from 90deg,
        transparent 0deg 292deg,
        rgba(167, 139, 250, 0.2) 318deg,
        transparent 342deg
    );
    opacity: 0.95;
    mix-blend-mode: screen;
    animation: radar-rotate 8s linear infinite;
}

.hero-node {
    position: absolute;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 8px rgba(167, 139, 250, 0.1), 0 0 18px rgba(167, 139, 250, 0.3);
    animation: node-pulse 2.8s ease-in-out infinite;
}

.hero-node.core {
    top: 48%;
    left: 51%;
    width: 16px;
    height: 16px;
    background: var(--accent-2);
    box-shadow: 0 0 0 9px rgba(45, 212, 191, 0.1), 0 0 24px rgba(45, 212, 191, 0.32);
    transform: translate(-50%, -50%);
}

.hero-node.a {
    top: 24%;
    left: 35%;
}

.hero-node.b {
    top: 30%;
    right: 26%;
    animation-delay: 0.4s;
}

.hero-node.c {
    bottom: 24%;
    left: 28%;
    animation-delay: 1.1s;
}

.hero-node.d {
    bottom: 28%;
    right: 30%;
    animation-delay: 1.7s;
}

.hero-link {
    position: absolute;
    height: 2px;
    border-radius: 999px;
    transform-origin: left center;
    background: linear-gradient(90deg, rgba(167, 139, 250, 0.1), rgba(167, 139, 250, 0.82), rgba(45, 212, 191, 0.2));
    box-shadow: 0 0 10px rgba(167, 139, 250, 0.2);
    opacity: 0.82;
}

.hero-link.a {
    top: 32%;
    left: 37%;
    width: 88px;
    transform: rotate(28deg);
}

.hero-link.b {
    top: 39%;
    right: 31%;
    width: 74px;
    transform: rotate(118deg);
}

.hero-link.c {
    bottom: 34%;
    left: 32%;
    width: 78px;
    transform: rotate(-34deg);
}

.hero-link.d {
    bottom: 36%;
    right: 34%;
    width: 68px;
    transform: rotate(208deg);
}

.hero-radar-caption {
    position: absolute;
    left: 16px;
    right: 16px;
    bottom: 14px;
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: end;
}

.hero-radar-copy strong {
    display: block;
    color: var(--ink);
    font-size: 18px;
    margin-bottom: 5px;
}

.hero-radar-copy p {
    margin: 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
    max-width: 320px;
}

.hero-radar-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 132px;
    padding: 10px 14px;
    border-radius: 18px;
    background: var(--panel-strong);
    border: 1px solid rgba(244, 114, 182, 0.2);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
    color: var(--ink);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

.hero-mini-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 14px;
}

.hero-mini-card {
    padding: 12px 12px 10px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
}

.hero-mini-card:hover {
    transform: translateY(-3px);
    border-color: rgba(167, 139, 250, 0.18);
    background: rgba(167, 139, 250, 0.08);
}

.hero-mini-card span {
    display: block;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 5px;
}

.hero-mini-card strong {
    display: block;
    color: var(--ink);
    font-size: 14px;
    line-height: 1.3;
}

.status-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 20px;
}

.status-pill {
    display: grid;
    grid-template-columns: 14px 1fr;
    gap: 10px;
    align-items: start;
    padding: 14px 14px 12px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
}

.status-pill:hover {
    transform: translateY(-3px);
    border-color: rgba(167, 139, 250, 0.16);
    background: rgba(167, 139, 250, 0.07);
}

.status-pill strong {
    display: block;
    color: var(--ink);
    font-size: 13px;
    margin-bottom: 3px;
}

.status-pill span:last-child {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
}

.control-card,
.insight-card,
.plot-shell,
.summary-shell,
.signal-grid-shell {
    position: relative;
    overflow: hidden;
    padding: 20px 20px 16px;
}

.control-card {
    animation-delay: 0.10s;
}

.insight-card {
    animation-delay: 0.16s;
}

.plot-shell {
    animation-delay: 0.24s;
}

.summary-shell {
    animation-delay: 0.30s;
}

.signal-grid-shell {
    animation-delay: 0.22s;
    padding-bottom: 16px;
    margin-bottom: 16px;
}

.control-card::before,
.insight-card::before,
.plot-shell::before,
.summary-shell::before,
.signal-grid-shell::before,
.comparison-band::before,
.metric-card::before {
    content: "";
    position: absolute;
    top: -140%;
    left: -35%;
    width: 42%;
    height: 320%;
    pointer-events: none;
    opacity: 0;
    transform: rotate(18deg);
    background: linear-gradient(
        90deg,
        rgba(255, 255, 255, 0),
        rgba(255, 255, 255, 0.08),
        rgba(255, 255, 255, 0)
    );
}

.control-card:hover::before,
.insight-card:hover::before,
.plot-shell:hover::before,
.summary-shell:hover::before,
.signal-grid-shell:hover::before,
.comparison-band:hover::before,
.metric-card:hover::before {
    opacity: 1;
    animation: shine-sweep 1.1s ease;
}

.panel-kicker {
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
}

.panel-title {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 24px;
    line-height: 1.15;
}

.panel-copy {
    margin: 0 0 14px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.insight-stack {
    display: grid;
    gap: 12px;
}

.insight-item {
    display: grid;
    grid-template-columns: 44px 1fr;
    gap: 12px;
    align-items: start;
    padding: 12px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition:
        transform 0.24s ease,
        border-color 0.24s ease,
        background 0.24s ease,
        box-shadow 0.24s ease;
}

.insight-index {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.2), rgba(45, 212, 191, 0.16));
    color: var(--ink);
    font-weight: 800;
    letter-spacing: 0.08em;
    transition: transform 0.24s ease, box-shadow 0.24s ease;
}

.insight-item:hover {
    transform: translateX(6px);
    background: rgba(167, 139, 250, 0.06);
    border-color: rgba(167, 139, 250, 0.18);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.16);
}

.insight-item:hover .insight-index {
    transform: scale(1.06);
    box-shadow: 0 10px 20px rgba(12, 30, 62, 0.22);
}

.insight-body strong {
    display: block;
    color: var(--ink);
    font-size: 14px;
    margin-bottom: 4px;
}

.insight-body p {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.comparison-band {
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: center;
    padding: 18px 22px;
    margin: 6px 0 16px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: var(--panel-strong);
    box-shadow: var(--shadow-soft);
    animation: rise-in 0.8s ease backwards;
    transition: transform 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease;
}

.comparison-band::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent);
    background-size: 180% 100%;
    animation: verdict-line 6s linear infinite;
}

.comparison-band:hover {
    transform: translateY(-4px);
    box-shadow: 0 26px 58px rgba(0, 0, 0, 0.24);
}

.comparison-band.win {
    border-color: rgba(45, 212, 191, 0.22);
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22), 0 0 0 1px rgba(45, 212, 191, 0.05);
}

.comparison-band.warn {
    border-color: rgba(251, 113, 133, 0.20);
}

.comparison-band.idle {
    border-color: rgba(167, 139, 250, 0.18);
}

.band-kicker {
    color: var(--accent);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 6px;
}

.band-title {
    color: var(--ink);
    font-size: 28px;
    line-height: 1.05;
    font-weight: 850;
    margin-bottom: 6px;
}

.band-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    max-width: 760px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 4px 0 8px;
}

.metric-card {
    position: relative;
    overflow: hidden;
    padding: 18px 18px 16px;
    min-height: 140px;
    background: linear-gradient(180deg, rgba(20, 18, 34, 0.86) 0%, rgba(12, 10, 24, 0.78) 100%);
    transition:
        transform 0.26s ease,
        border-color 0.26s ease,
        box-shadow 0.26s ease,
        background 0.26s ease;
}

.metric-card::after {
    content: "";
    position: absolute;
    inset: auto -40px -50px auto;
    width: 130px;
    height: 130px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.14), transparent 70%);
    pointer-events: none;
    transition: transform 0.28s ease, opacity 0.28s ease;
}

.metric-card.spotlight {
    border-color: rgba(167, 139, 250, 0.24);
    box-shadow: 0 22px 56px rgba(0, 0, 0, 0.24), 0 0 0 1px rgba(167, 139, 250, 0.05);
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: rgba(167, 139, 250, 0.24);
    box-shadow: 0 22px 48px rgba(0, 0, 0, 0.22), 0 0 0 1px rgba(167, 139, 250, 0.05);
}

.metric-card:hover::after {
    transform: scale(1.18);
    opacity: 0.96;
}

.metric-label {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
    color: var(--ink);
}

.metric-value.win {
    color: var(--accent-2);
}

.metric-value.warn {
    color: #ffd27d;
}

.metric-subtext {
    margin-top: 8px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.5;
}

.signal-shell-header {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 16px;
    margin-bottom: 16px;
}

.signal-title {
    margin: 0;
    color: var(--ink);
    font-size: 24px;
}

.signal-copy {
    margin: 6px 0 0;
    max-width: 660px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.65;
}

.signal-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.signal-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: var(--ink);
    font-size: 12px;
    font-weight: 700;
}

.signal-tag::before {
    content: "";
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px rgba(167, 139, 250, 0.34);
}

.signal-tag.random::before {
    background: var(--warn);
    box-shadow: 0 0 12px rgba(251, 113, 133, 0.3);
}

.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}

.signal-row {
    position: relative;
    overflow: hidden;
    padding: 16px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.07);
    transition: transform 0.24s ease, border-color 0.24s ease, background 0.24s ease, box-shadow 0.24s ease;
}

.signal-row:hover {
    transform: translateY(-5px);
    border-color: rgba(167, 139, 250, 0.18);
    background: rgba(167, 139, 250, 0.06);
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.18);
}

.signal-row.highlight {
    border-color: rgba(167, 139, 250, 0.18);
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18), 0 0 0 1px rgba(167, 139, 250, 0.05);
}

.signal-row-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 12px;
    margin-bottom: 10px;
}

.signal-row-label {
    color: var(--ink);
    font-size: 15px;
    font-weight: 800;
}

.signal-row-copy {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
}

.signal-row-delta {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px 10px;
    border-radius: 14px;
    background: rgba(167, 139, 250, 0.1);
    color: var(--ink);
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

.signal-row-delta.warn {
    background: rgba(251, 113, 133, 0.1);
}

.signal-lane {
    display: grid;
    grid-template-columns: 58px minmax(0, 1fr) auto;
    gap: 10px;
    align-items: center;
    margin-top: 10px;
}

.signal-lane-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.signal-track {
    position: relative;
    overflow: hidden;
    height: 11px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.signal-track::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.04), transparent 60%);
}

.signal-fill {
    position: relative;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 55%, #2dd4bf 100%);
    box-shadow: 0 0 18px rgba(167, 139, 250, 0.22);
}

.signal-fill::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.24), transparent);
    transform: translateX(-110%);
    animation: signal-scan 2.8s ease-in-out infinite;
}

.signal-fill.random {
    background: linear-gradient(90deg, #fb7185 0%, #f9a8d4 55%, #fbbf24 100%);
    box-shadow: 0 0 18px rgba(251, 113, 133, 0.2);
}

.signal-fill.neutral {
    background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 100%);
    box-shadow: 0 0 18px rgba(244, 114, 182, 0.2);
}

.signal-lane-value {
    color: var(--ink);
    font-size: 12px;
    font-weight: 700;
    min-width: 56px;
    text-align: right;
}

.signal-foot {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
}

.summary-shell {
    padding-bottom: 12px;
}

.summary-title {
    margin: 0 0 8px;
    font-size: 24px;
    color: var(--ink);
}

.summary-copy {
    margin: 0 0 12px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.score-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 14px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.18), rgba(45, 212, 191, 0.12));
    border: 1px solid rgba(255, 255, 255, 0.09);
    color: var(--ink);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    white-space: nowrap;
    transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
}

.comparison-band:hover .score-pill {
    transform: translateY(-2px);
    border-color: rgba(45, 212, 191, 0.18);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
}

.plot-title {
    margin: 0 0 10px;
    font-size: 22px;
    color: var(--ink);
}

.plot-copy {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
}

.log-panel textarea {
    min-height: 330px !important;
    border-radius: 18px !important;
    border: 1px solid rgba(110, 163, 255, 0.18) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    transition: border-color 0.24s ease, box-shadow 0.24s ease, transform 0.24s ease !important;
}

.log-panel textarea:hover,
.log-panel textarea:focus {
    border-color: rgba(167, 139, 250, 0.28) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 0 0 3px rgba(167, 139, 250, 0.08) !important;
}

.gr-button-primary {
    position: relative;
    overflow: hidden !important;
    background: linear-gradient(135deg, var(--button-a), var(--button-b)) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 16px 34px rgba(124, 58, 237, 0.34) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    isolation: isolate;
}

.gr-button-primary::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    transform: translateX(-120%) skewX(-18deg);
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.24), transparent);
}

.gr-button-primary:hover {
    filter: brightness(1.05);
    transform: translateY(-2px);
    box-shadow: 0 18px 38px rgba(167, 139, 250, 0.34) !important;
}

.gr-button-primary:hover::before {
    animation: button-sheen 0.9s ease;
}

.gr-button-primary:active {
    transform: translateY(0) scale(0.992);
}

#run-deck-button {
    position: relative;
    isolation: isolate;
}

#run-deck-button::after {
    content: "";
    position: absolute;
    inset: -10px;
    border-radius: inherit;
    border: 1px solid rgba(167, 139, 250, 0.12);
    opacity: 0;
    animation: button-pulse 2.8s ease-out infinite;
}

.gr-slider input[type="range"] {
    accent-color: var(--accent);
    transition: filter 0.22s ease, transform 0.22s ease;
}

.control-card .gr-slider .wrap {
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    background: rgba(255, 255, 255, 0.035) !important;
    padding: 12px 14px !important;
}

.control-card .gr-slider label {
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 12px !important;
}

.control-card:hover .gr-slider input[type="range"] {
    filter: drop-shadow(0 0 8px rgba(167, 139, 250, 0.28));
}

.control-footer {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.control-stat {
    padding: 12px 14px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
}

.control-stat:hover {
    transform: translateY(-3px);
    border-color: rgba(167, 139, 250, 0.18);
    background: rgba(167, 139, 250, 0.07);
}

.control-stat span {
    display: block;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 5px;
}

.control-stat strong {
    display: block;
    color: var(--ink);
    font-size: 14px;
    line-height: 1.35;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-group {
    border-color: var(--panel-border) !important;
    background: transparent !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap {
    background: var(--input-background-fill) !important;
    color: var(--ink) !important;
    border-color: var(--input-border-color) !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    background: var(--input-background-fill-focus) !important;
    border-color: var(--accent) !important;
}

.gradio-container input[type="number"],
.gradio-container .gr-number input {
    background: var(--input-background-fill) !important;
    color: var(--ink) !important;
    border-color: var(--input-border-color) !important;
}

.gradio-container .gr-slider .wrap,
.gradio-container .slider_input,
.gradio-container .range_input {
    background: transparent !important;
}

.gradio-container input[type="range"] {
    background: transparent !important;
}

.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: rgba(167, 139, 250, 0.2) !important;
}

.gradio-container input[type="range"]::-webkit-slider-thumb {
    background: #a78bfa !important;
}

.gradio-container .block.padded,
.gradio-container .gr-padded {
    background: transparent !important;
}

.gradio-container .label-wrap span,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p {
    color: var(--ink) !important;
}

.gradio-container .prose strong {
    color: var(--ink) !important;
}

.control-card .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.plot-shell .plot-container,
.plot-shell .plot-container > div {
    min-height: 480px !important;
}

* {
    scrollbar-width: thin;
    scrollbar-color: rgba(167, 139, 250, 0.5) rgba(12, 12, 20, 0.5);
}

*::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

*::-webkit-scrollbar-track {
    background: var(--panel);
}

*::-webkit-scrollbar-thumb {
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(167, 139, 250, 0.7), rgba(45, 212, 191, 0.5));
    border: 2px solid var(--panel);
}

@keyframes rise-in {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes ambient-drift {
    0% {
        transform: translate3d(-1.5%, -1%, 0) scale(1);
        opacity: 0.56;
    }
    100% {
        transform: translate3d(1.5%, 1.5%, 0) scale(1.08);
        opacity: 0.78;
    }
}

@keyframes orb-drift-a {
    0% {
        transform: translate3d(0, 0, 0) scale(1);
    }
    100% {
        transform: translate3d(16px, 12px, 0) scale(1.08);
    }
}

@keyframes orb-drift-b {
    0% {
        transform: translate3d(0, 0, 0) scale(1);
    }
    100% {
        transform: translate3d(-14px, -10px, 0) scale(1.06);
    }
}

@keyframes chip-float {
    0%,
    100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-3px);
    }
}

@keyframes shine-sweep {
    0% {
        transform: translateX(-160%) rotate(18deg);
    }
    100% {
        transform: translateX(320%) rotate(18deg);
    }
}

@keyframes verdict-line {
    0% {
        background-position: 180% 0;
    }
    100% {
        background-position: -20% 0;
    }
}

@keyframes button-sheen {
    0% {
        transform: translateX(-120%) skewX(-18deg);
    }
    100% {
        transform: translateX(180%) skewX(-18deg);
    }
}

@keyframes radar-rotate {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes radar-pulse {
    0%,
    100% {
        opacity: 0.26;
        transform: scale(1);
    }
    50% {
        opacity: 0.68;
        transform: scale(1.035);
    }
}

@keyframes node-pulse {
    0%,
    100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.18);
    }
}

@keyframes beacon-pulse {
    0%,
    100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.82;
        transform: scale(1.15);
    }
}

@keyframes signal-scan {
    0% {
        transform: translateX(-110%);
    }
    100% {
        transform: translateX(180%);
    }
}

@keyframes button-pulse {
    0% {
        opacity: 0.72;
        transform: scale(0.97);
    }
    100% {
        opacity: 0;
        transform: scale(1.08);
    }
}

@keyframes hero-sheen {
    0%,
    72%,
    100% {
        transform: translateX(-100%);
    }
    84% {
        transform: translateX(130%);
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 34px;
    }

    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .comparison-band {
        flex-direction: column;
        align-items: flex-start;
    }

    .status-strip,
    .control-footer,
    .signal-grid {
        grid-template-columns: 1fr;
    }

    .signal-shell-header {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (max-width: 640px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .hero-panel,
    .control-card,
    .insight-card,
    .summary-shell,
    .metric-card,
    .plot-shell,
    .comparison-band,
    .signal-grid-shell {
        border-radius: 16px;
    }

    .hero-panel {
        padding: 28px 24px 24px;
    }

    .hero-mini-grid {
        grid-template-columns: 1fr;
    }

    .hero-radar-caption {
        flex-direction: column;
        align-items: flex-start;
    }
}

.hero-bigstat-strip {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 24px 0 20px;
    padding: 20px 22px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.08), rgba(45, 212, 191, 0.06));
    border: 1px solid rgba(167, 139, 250, 0.14);
}

.hero-bigstat {
    flex: 1;
    text-align: center;
}

.hero-bigstat-num {
    display: block;
    font-size: 52px;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, #a78bfa 0%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}

.hero-bigstat-label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    line-height: 1.4;
}

.hero-bigstat-divider {
    width: 1px;
    height: 48px;
    background: linear-gradient(180deg, transparent, rgba(167, 139, 250, 0.3), transparent);
    margin: 0 8px;
}

.hero-feature-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 20px;
}

.hero-feature-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: background 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
}

.hero-feature-item:hover {
    background: rgba(167, 139, 250, 0.07);
    border-color: rgba(167, 139, 250, 0.16);
    transform: translateX(4px);
}

.hero-feature-dot {
    flex-shrink: 0;
    margin-top: 4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #a78bfa;
    box-shadow: 0 0 10px rgba(167, 139, 250, 0.5);
}

.hero-feature-dot.teal {
    background: #2dd4bf;
    box-shadow: 0 0 10px rgba(45, 212, 191, 0.5);
}

.hero-feature-dot.amber {
    background: #fbbf24;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
}

.hero-feature-item strong {
    display: block;
    color: var(--ink);
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 2px;
}

.hero-feature-item span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
}
"""

HERO_HTML = """
<div class="app-shell">
  <section class="hero-panel">
    <div class="hero-grid">
      <div class="hero-copy-block">
        <div class="hero-brow">Reinforcement learning command deck</div>
        <h1 class="hero-title">Supply Chain RL <span class="hero-accent">Control Room</span></h1>
        <p class="hero-copy">
          Stress-test the trained control policy against a random decision-maker under the exact same shocks,
          demand spikes, and supplier delays. This dashboard is built to feel like an executive control surface,
          not just a model demo.
        </p>
        <div class="hero-bigstat-strip">
          <div class="hero-bigstat">
            <span class="hero-bigstat-num">5</span>
            <span class="hero-bigstat-label">Matched<br>Seeds</span>
          </div>
          <div class="hero-bigstat-divider"></div>
          <div class="hero-bigstat">
            <span class="hero-bigstat-num">100</span>
            <span class="hero-bigstat-label">Steps per<br>Episode</span>
          </div>
          <div class="hero-bigstat-divider"></div>
          <div class="hero-bigstat">
            <span class="hero-bigstat-num">2</span>
            <span class="hero-bigstat-label">Competing<br>Agents</span>
          </div>
        </div>
        <div class="hero-feature-list">
          <div class="hero-feature-item">
            <span class="hero-feature-dot"></span>
            <div>
              <strong>Fair matchup</strong>
              <span>Identical disruptions, demand spikes &amp; supplier delays for both agents</span>
            </div>
          </div>
          <div class="hero-feature-item">
            <span class="hero-feature-dot teal"></span>
            <div>
              <strong>Deterministic policy replay</strong>
              <span>Trained controller runs without action noise — clean, reproducible signal</span>
            </div>
          </div>
          <div class="hero-feature-item">
            <span class="hero-feature-dot amber"></span>
            <div>
              <strong>Averaged scoreboard</strong>
              <span>Results are averaged across 5 rollouts to filter out single-episode luck</span>
            </div>
          </div>
        </div>
        <div class="hero-chip-row">
          <span class="hero-chip">Matched seeded scenarios</span>
          <span class="hero-chip">Deterministic trained policy</span>
          <span class="hero-chip">Reward, fulfillment &amp; stockout tracking</span>
        </div>
      </div>
      <div class="hero-side-stack">
        <div class="hero-visual-shell">
          <div class="hero-visual-top">
            <div class="hero-visual-label"><span class="status-dot alt"></span>Live network pulse</div>
            <div class="hero-visual-status"><span class="status-dot"></span>Fair-match protocol online</div>
          </div>
          <div class="hero-radar">
            <div class="hero-sweep"></div>
            <div class="hero-ring a"></div>
            <div class="hero-ring b"></div>
            <div class="hero-ring c"></div>
            <span class="hero-node core"></span>
            <span class="hero-node a"></span>
            <span class="hero-node b"></span>
            <span class="hero-node c"></span>
            <span class="hero-node d"></span>
            <span class="hero-link a"></span>
            <span class="hero-link b"></span>
            <span class="hero-link c"></span>
            <span class="hero-link d"></span>
            <div class="hero-radar-caption">
              <div class="hero-radar-copy">
                <strong>Control surface primed</strong>
                <p>One scenario, two policies, synchronized disruptions, and a fast visual read on who stays steady under pressure.</p>
              </div>
              <div class="hero-radar-badge">5 sync runs</div>
            </div>
          </div>
          <div class="hero-mini-grid">
            <div class="hero-mini-card">
              <span>Signal</span>
              <strong>Demand volatility mapped</strong>
            </div>
            <div class="hero-mini-card">
              <span>Focus</span>
              <strong>Service level with cost discipline</strong>
            </div>
            <div class="hero-mini-card">
              <span>Mode</span>
              <strong>Executive comparison deck</strong>
            </div>
          </div>
        </div>
        <div class="hero-statboard">
          <div class="hero-stat">
            <span class="hero-stat-label">Evaluation mode</span>
            <span class="hero-stat-value">5 synchronized rollouts</span>
            <span class="hero-stat-copy">
              Both agents face the same turbulence so the result reflects decision quality rather than lucky randomness.
            </span>
          </div>
          <div class="hero-stat">
            <span class="hero-stat-label">Primary outcome</span>
            <span class="hero-stat-value">Service level with discipline</span>
            <span class="hero-stat-copy">
              The strongest policy keeps fulfillment high while containing stockouts and preserving total reward.
            </span>
          </div>
        </div>
      </div>
    </div>
    <div class="status-strip">
      <div class="status-pill">
        <span class="status-dot"></span>
        <span>
          <strong>Matched turbulence</strong>
          Demand spikes and supplier delays are mirrored across both lanes.
        </span>
      </div>
      <div class="status-pill">
        <span class="status-dot alt"></span>
        <span>
          <strong>Deterministic trained agent</strong>
          The learned policy replays without action noise so comparisons stay readable.
        </span>
      </div>
      <div class="status-pill">
        <span class="status-dot warn"></span>
        <span>
          <strong>Fast decision feedback</strong>
          Charts, verdict cards, and the transcript update together after every launch.
        </span>
      </div>
    </div>
  </section>
</div>
"""

DEFAULT_HIGHLIGHTS = """
<div class="app-shell">
  <section class="comparison-band idle">
    <div>
      <div class="band-kicker">Awaiting simulation</div>
      <div class="band-title">Shape the network, then light up the deck.</div>
      <div class="band-copy">
        The result board will spotlight the winning policy, reward edge, service-level lift, stockout reduction, and checkpoint in play.
      </div>
    </div>
    <div class="score-pill">5 matched episodes / 100 steps each</div>
  </section>
  <section class="signal-grid-shell">
    <div class="signal-shell-header">
      <div>
        <div class="panel-kicker">Signal board</div>
        <h3 class="signal-title">Telemetry wakes up after the first launch.</h3>
        <p class="signal-copy">
          The animated meter lanes will compare service, resilience, and reward momentum for the trained and random policies side by side.
        </p>
      </div>
      <div class="signal-legend">
        <span class="signal-tag">Trained lane</span>
        <span class="signal-tag random">Random lane</span>
      </div>
    </div>
    <div class="signal-grid">
      <div class="signal-row highlight">
        <div class="signal-row-header">
          <div>
            <div class="signal-row-label">Service level</div>
            <p class="signal-row-copy">Placeholder bands preview the visual treatment before the first run.</p>
          </div>
          <div class="signal-row-delta">Stand by</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Trained</div>
          <div class="signal-track"><div class="signal-fill" style="width: 78%;"></div></div>
          <div class="signal-lane-value">78%</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Random</div>
          <div class="signal-track"><div class="signal-fill random" style="width: 56%;"></div></div>
          <div class="signal-lane-value">56%</div>
        </div>
        <div class="signal-foot">Run the simulation to replace these placeholders with the real match result.</div>
      </div>
      <div class="signal-row">
        <div class="signal-row-header">
          <div>
            <div class="signal-row-label">Resilience</div>
            <p class="signal-row-copy">Higher bars mean fewer stockouts and steadier network recovery.</p>
          </div>
          <div class="signal-row-delta">Queued</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Trained</div>
          <div class="signal-track"><div class="signal-fill neutral" style="width: 71%;"></div></div>
          <div class="signal-lane-value">71%</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Random</div>
          <div class="signal-track"><div class="signal-fill random" style="width: 49%;"></div></div>
          <div class="signal-lane-value">49%</div>
        </div>
        <div class="signal-foot">The live version will derive this directly from the averaged stockout rate.</div>
      </div>
      <div class="signal-row">
        <div class="signal-row-header">
          <div>
            <div class="signal-row-label">Reward momentum</div>
            <p class="signal-row-copy">This lane turns total reward into an instant visual comparison.</p>
          </div>
          <div class="signal-row-delta warn">Awaiting data</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Trained</div>
          <div class="signal-track"><div class="signal-fill" style="width: 74%;"></div></div>
          <div class="signal-lane-value">74%</div>
        </div>
        <div class="signal-lane">
          <div class="signal-lane-label">Random</div>
          <div class="signal-track"><div class="signal-fill random" style="width: 52%;"></div></div>
          <div class="signal-lane-value">52%</div>
        </div>
        <div class="signal-foot">The verdict band above will flip from idle to live once a run completes.</div>
      </div>
    </div>
  </section>
</div>
"""

GUIDE_PANEL_HTML = """
<div class="insight-card">
  <div class="panel-kicker">Evaluation protocol</div>
  <h3 class="panel-title">One network. Two decision-makers. Same turbulence.</h3>
  <p class="panel-copy">
    The visual comparison is only useful if the simulation is fair. This deck keeps the matchup disciplined.
  </p>
  <div class="insight-stack">
    <div class="insight-item">
      <div class="insight-index">01</div>
      <div class="insight-body">
        <strong>Matched disruptions</strong>
        <p>Demand spikes and supplier delays are synchronized across both agents.</p>
      </div>
    </div>
    <div class="insight-item">
      <div class="insight-index">02</div>
      <div class="insight-body">
        <strong>Deterministic policy replay</strong>
        <p>The trained controller is evaluated without action noise so the signal stays clean.</p>
      </div>
    </div>
    <div class="insight-item">
      <div class="insight-index">03</div>
      <div class="insight-body">
        <strong>Averaged scoreboards</strong>
        <p>Charts summarize five full 100-step episodes to reduce single-rollout luck.</p>
      </div>
    </div>
  </div>
</div>
"""


def get_model_candidates(n_warehouses):
    model_map = {
        2: "agent_w2_r3",
        3: "agent_w3_r5",
        4: "agent_w4_r7",
        5: "agent_w5_r8",
        6: "agent_w6_r10",
    }
    base_name = model_map.get(int(n_warehouses), "supply_chain_agent")
    return [
        (f"{base_name}_a2c", A2C, "A2C"),
        (base_name, PPO, "PPO"),
        ("supply_chain_agent_a2c", A2C, "A2C"),
        ("supply_chain_agent", PPO, "PPO"),
    ]


def load_agent(n_warehouses):
    errors = []

    for model_path, model_cls, algorithm_name in get_model_candidates(n_warehouses):
        if not os.path.exists(model_path + ".zip"):
            continue
        try:
            return model_cls.load(model_path), model_path, algorithm_name
        except Exception as e:
            errors.append(f"{model_path}.zip ({algorithm_name}): {e}")

    if errors:
        print("Model load errors:")
        for error in errors:
            print(f"  - {error}")

    fallback_path, _, fallback_algorithm = get_model_candidates(n_warehouses)[0]
    return None, fallback_path, fallback_algorithm


def predict_agent_action(agent, obs, env):
    action, _ = agent.predict(obs, deterministic=True)

    agent_low = np.asarray(agent.action_space.low, dtype=np.float32)
    agent_high = np.asarray(agent.action_space.high, dtype=np.float32)
    env_low = np.asarray(env.action_space.low, dtype=np.float32)
    env_high = np.asarray(env.action_space.high, dtype=np.float32)

    if not (
        np.allclose(agent_low, env_low) and np.allclose(agent_high, env_high)
    ):
        action = env_low + (action - agent_low) * (env_high - env_low) / (
            agent_high - agent_low + 1e-9
        )

    return np.clip(action, env_low, env_high)


def run_episode(n_warehouses, n_retailers, seed, agent=None):
    env = SupplyChainEnv(
        n_warehouses=int(n_warehouses),
        n_retailers=int(n_retailers)
    )

    obs = env.reset(seed=seed)
    # Convert Pydantic Observation to numpy for agent
    obs_np = np.concatenate([obs.inventory, obs.demand, [obs.timestep / obs.max_steps]]).astype(np.float32)
    
    rewards, fulfillments, stockouts = [], [], []
    total_reward = 0

    for step in range(100):
        if agent is not None:
            # RL agent needs numpy array
            action_raw, _ = agent.predict(obs_np, deterministic=True)
            action = Action(restock_quantities=action_raw.tolist())
        else:
            action_raw = env.action_space.sample()
            action = Action(restock_quantities=action_raw.tolist())

        obs, reward, done, info = env.step(action)
        # Update obs_np for next step
        obs_np = np.concatenate([obs.inventory, obs.demand, [obs.timestep / obs.max_steps]]).astype(np.float32)
        
        total_reward += reward.value
        rewards.append(reward.value)
        fulfillments.append(info.fulfillment_rate * 100)
        stockouts.append(info.stockout_rate * 100)
        if done:
            break

    summary = env.get_episode_summary()
    return rewards, fulfillments, stockouts, summary, total_reward


def aggregate_runs(n_warehouses, n_retailers, seeds, agent=None):
    rewards_runs, fulfill_runs, stockout_runs = [], [], []
    summaries, total_rewards = [], []

    for seed in seeds:
        rewards, fulfillments, stockouts, summary, total_reward = run_episode(
            n_warehouses, n_retailers, seed=seed, agent=agent
        )
        rewards_runs.append(np.array(rewards, dtype=np.float32))
        fulfill_runs.append(np.array(fulfillments, dtype=np.float32))
        stockout_runs.append(np.array(stockouts, dtype=np.float32))
        summaries.append(summary)
        total_rewards.append(total_reward)

    mean_summary = {
        "total_steps": int(round(np.mean([s.get("total_steps", 0) for s in summaries]))),
        "avg_reward": round(float(np.mean([s.get("avg_reward", 0) for s in summaries])), 3),
        "fulfillment_rate_%": round(float(np.mean([s.get("fulfillment_rate_%", 0) for s in summaries])), 1),
        "stockout_rate_%": round(float(np.mean([s.get("stockout_rate_%", 0) for s in summaries])), 1),
        "total_demand": round(float(np.mean([s.get("total_demand", 0) for s in summaries])), 1),
        "total_fulfilled": round(float(np.mean([s.get("total_fulfilled", 0) for s in summaries])), 1),
    }

    return {
        "rewards": np.mean(np.stack(rewards_runs), axis=0),
        "fulfillments": np.mean(np.stack(fulfill_runs), axis=0),
        "stockouts": np.mean(np.stack(stockout_runs), axis=0),
        "summary": mean_summary,
        "total_reward": float(np.mean(total_rewards)),
    }


def metric_delta(trained_value, random_value, suffix="", invert=False):
    delta = trained_value - random_value
    if invert:
        delta = random_value - trained_value
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.1f}{suffix}"


def clamp_percent(value, min_value=0.0, max_value=100.0):
    return max(min_value, min(max_value, float(value)))


def reward_signal(value, reference):
    span = max(abs(value), abs(reference), 1.0)
    return clamp_percent(50 + (value / span) * 50)


def build_signal_row(label, copy, delta_text, tone, trained_width, trained_value, random_width, random_value, footnote, highlight=False):
    row_class = "signal-row highlight" if highlight else "signal-row"
    delta_class = "signal-row-delta warn" if tone == "warn" else "signal-row-delta"
    return f"""
    <div class="{row_class}">
      <div class="signal-row-header">
        <div>
          <div class="signal-row-label">{label}</div>
          <p class="signal-row-copy">{copy}</p>
        </div>
        <div class="{delta_class}">{delta_text}</div>
      </div>
      <div class="signal-lane">
        <div class="signal-lane-label">Trained</div>
        <div class="signal-track"><div class="signal-fill" style="width: {trained_width:.1f}%;"></div></div>
        <div class="signal-lane-value">{trained_value}</div>
      </div>
      <div class="signal-lane">
        <div class="signal-lane-label">Random</div>
        <div class="signal-track"><div class="signal-fill random" style="width: {random_width:.1f}%;"></div></div>
        <div class="signal-lane-value">{random_value}</div>
      </div>
      <div class="signal-foot">{footnote}</div>
    </div>
    """


def style_plot_axis(ax):
    ax.set_facecolor('none')
    ax.grid(axis='y', color='#94a3b8', alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#64748b', labelsize=8.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#94a3b8')
        ax.spines[spine].set_alpha(0.2)
        ax.spines[spine].set_linewidth(1.0)


def annotate_terminal(ax, x_value, y_value, text, color):
    ax.scatter([x_value], [y_value], color=color, s=34, zorder=5)
    ax.annotate(
        text,
        (x_value, y_value),
        xytext=(8, 0),
        textcoords='offset points',
        color='white',
        fontsize=8,
        va='center',
        bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none', alpha=0.9),
    )


def label_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        offset = 1.2 if height >= 0 else -3.0
        va = 'bottom' if height >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f'{height:.1f}',
            ha='center',
            va=va,
            color='#f8fafc',
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.22', fc='#475569', ec='none', alpha=0.6),
        )


def build_highlights_html(
    n_warehouses,
    n_retailers,
    model_name,
    algorithm_name,
    trained_agent_loaded,
    t_summary,
    r_summary,
    t_total,
    r_total,
    improvement,
    num_seeds,
):
    result_label = "Trained policy is in command" if t_total >= r_total else "Random baseline steals the lead"
    result_tone = "win" if t_total >= r_total else "warn"
    model_note = (
        f"Loaded {algorithm_name} checkpoint {model_name}.zip"
        if trained_agent_loaded
        else f"No trained {algorithm_name} checkpoint was available, so the trained lane fell back to random actions."
    )
    scenario_note = (
        f"{n_warehouses} warehouses / {n_retailers} retailers / "
        f"{num_seeds} matched episodes / 100 steps each"
    )
    t_fulfillment = t_summary.get("fulfillment_rate_%", 0)
    r_fulfillment = r_summary.get("fulfillment_rate_%", 0)
    t_stockout = t_summary.get("stockout_rate_%", 0)
    r_stockout = r_summary.get("stockout_rate_%", 0)
    t_resilience = clamp_percent(100 - t_stockout)
    r_resilience = clamp_percent(100 - r_stockout)
    t_reward_signal = reward_signal(t_total, r_total)
    r_reward_signal = reward_signal(r_total, t_total)

    signal_rows_html = "".join(
        [
            build_signal_row(
                label="Service level",
                copy="Higher service means the network is meeting retailer demand more consistently.",
                delta_text=metric_delta(t_fulfillment, r_fulfillment, suffix=" pts"),
                tone="win" if t_fulfillment >= r_fulfillment else "warn",
                trained_width=clamp_percent(t_fulfillment),
                trained_value=f"{t_fulfillment:.1f}%",
                random_width=clamp_percent(r_fulfillment),
                random_value=f"{r_fulfillment:.1f}%",
                footnote="This is the cleanest read on whether the policy protects downstream service under turbulence.",
                highlight=True,
            ),
            build_signal_row(
                label="Network resilience",
                copy="This lane converts stockout rate into a higher-is-better resilience signal.",
                delta_text=metric_delta(t_stockout, r_stockout, suffix=" pts", invert=True),
                tone="win" if t_stockout <= r_stockout else "warn",
                trained_width=t_resilience,
                trained_value=f"{t_resilience:.1f}%",
                random_width=r_resilience,
                random_value=f"{r_resilience:.1f}%",
                footnote="Lower stockout rate creates a taller resilience bar and a steadier operating profile.",
            ),
            build_signal_row(
                label="Reward momentum",
                copy="Total reward is normalized into an instant visual pulse so the winner stands out fast.",
                delta_text=f"{improvement:+.1f}%",
                tone=result_tone,
                trained_width=t_reward_signal,
                trained_value=f"{t_total:.1f}",
                random_width=r_reward_signal,
                random_value=f"{r_total:.1f}",
                footnote=f"{algorithm_name} checkpoint {model_name}.zip is driving the trained lane." if trained_agent_loaded else model_note,
            ),
        ]
    )

    cards = [
        (
            "Reward edge",
            f"{improvement:+.1f}%",
            f"Trained {t_total:.1f} total reward vs random {r_total:.1f}.",
            result_tone,
            "spotlight",
        ),
        (
            "Fulfillment lift",
            metric_delta(
                t_summary.get("fulfillment_rate_%", 0),
                r_summary.get("fulfillment_rate_%", 0),
                suffix=" pts",
            ),
            f"Trained {t_summary.get('fulfillment_rate_%', 0):.1f}% vs random {r_summary.get('fulfillment_rate_%', 0):.1f}%.",
            "win",
            "",
        ),
        (
            "Stockout reduction",
            metric_delta(
                t_summary.get("stockout_rate_%", 0),
                r_summary.get("stockout_rate_%", 0),
                suffix=" pts",
                invert=True,
            ),
            f"Lower is better. Trained {t_summary.get('stockout_rate_%', 0):.1f}% vs random {r_summary.get('stockout_rate_%', 0):.1f}%.",
            "win",
            "",
        ),
        (
            "Active checkpoint",
            model_name,
            f"{model_note}",
            "win" if trained_agent_loaded else "warn",
            "",
        ),
    ]

    cards_html = "".join(
        f"""
        <div class="metric-card {extra_class}">
          <div class="metric-label">{label}</div>
          <div class="metric-value {tone}">{value}</div>
          <div class="metric-subtext">{subtext}</div>
        </div>
        """
        for label, value, subtext, tone, extra_class in cards
    )

    return f"""
    <div class="app-shell">
      <section class="comparison-band {result_tone}">
        <div>
          <div class="band-kicker">Live verdict</div>
          <div class="band-title">{result_label}</div>
          <div class="band-copy">{scenario_note}</div>
        </div>
        <div class="score-pill">{algorithm_name} / {model_name}.zip</div>
      </section>
      <section class="signal-grid-shell">
        <div class="signal-shell-header">
          <div>
            <div class="panel-kicker">Signal board</div>
            <h3 class="signal-title">Policy telemetry is now live.</h3>
            <p class="signal-copy">
              These animated lanes turn the averaged evaluation metrics into a faster scan: service quality, resilience under pressure, and reward momentum.
            </p>
          </div>
          <div class="signal-legend">
            <span class="signal-tag">Trained lane</span>
            <span class="signal-tag random">Random lane</span>
          </div>
        </div>
        <div class="signal-grid">
          {signal_rows_html}
        </div>
      </section>
      <div class="metric-grid">
        {cards_html}
      </div>
    </div>
    """


def run_demo(n_warehouses, n_retailers):
    n_warehouses = int(n_warehouses)
    n_retailers  = int(n_retailers)
    seeds = EVAL_SEEDS
    trained_agent, model_name, algorithm_name = load_agent(n_warehouses)

    trained_results = aggregate_runs(
        n_warehouses, n_retailers, seeds=seeds, agent=trained_agent
    )
    random_results = aggregate_runs(
        n_warehouses, n_retailers, seeds=seeds, agent=None
    )

    t_rewards = trained_results["rewards"]
    t_fulfill = trained_results["fulfillments"]
    t_summary = trained_results["summary"]
    t_total = trained_results["total_reward"]

    r_rewards = random_results["rewards"]
    r_fulfill = random_results["fulfillments"]
    r_summary = random_results["summary"]
    r_total = random_results["total_reward"]

    steps = list(range(1, len(t_rewards) + 1))
    trained_color = '#a78bfa'
    trained_fill  = '#7c3aed'
    random_color  = '#fb7185'
    random_fill   = '#be123c'
    fulfillment_color = '#2dd4bf'

    fig1, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=130)
    fig1.patch.set_facecolor('none')
    for ax in axes:
        style_plot_axis(ax)

    axes[0].plot(
        steps, t_rewards, color=trained_color, linewidth=2.7, label='Trained agent',
        solid_capstyle='round'
    )
    axes[0].plot(
        steps, r_rewards, color=random_color, linewidth=2.2, label='Random agent',
        alpha=0.82, solid_capstyle='round'
    )
    axes[0].fill_between(steps, t_rewards, 0, color=trained_fill, alpha=0.16)
    axes[0].fill_between(steps, r_rewards, 0, color=random_fill, alpha=0.10)
    axes[0].axhline(0, color='#e2e8f0', linestyle='--', alpha=0.22, linewidth=1)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title('Reward trajectory', color='#64748b', fontsize=14, fontweight='bold', loc='left')
    axes[0].text(
        0.02, 0.92, 'Higher is better',
        transform=axes[0].transAxes, color='#94a3b8', fontsize=8.5
    )
    axes[0].set_xlabel('Step', color='#64748b')
    axes[0].set_ylabel('Reward signal', color='#64748b')
    axes[0].legend(
        facecolor='none',
        edgecolor='none',
        labelcolor='#64748b',
        framealpha=0.0,
        fontsize=8.5,
        loc='lower left',
    )
    annotate_terminal(axes[0], steps[-1], t_rewards[-1], f"{t_rewards[-1]:.2f}", trained_color)
    annotate_terminal(axes[0], steps[-1], r_rewards[-1], f"{r_rewards[-1]:.2f}", random_color)

    axes[1].plot(
        steps, t_fulfill, color=fulfillment_color, linewidth=2.7, label='Trained agent',
        solid_capstyle='round'
    )
    axes[1].plot(
        steps, r_fulfill, color=random_color, linewidth=2.2, label='Random agent',
        alpha=0.82, solid_capstyle='round'
    )
    axes[1].fill_between(steps, t_fulfill, color='#0d9488', alpha=0.15)
    axes[1].fill_between(steps, r_fulfill, color=random_fill, alpha=0.10)
    axes[1].set_ylim(0, 105)
    axes[1].set_title('Service level curve', color='#64748b', fontsize=14, fontweight='bold', loc='left')
    axes[1].text(
        0.02, 0.92, 'Average fulfillment across matched episodes',
        transform=axes[1].transAxes, color='#94a3b8', fontsize=8.5
    )
    axes[1].set_xlabel('Step', color='#64748b')
    axes[1].set_ylabel('Fulfillment %', color='#64748b')
    axes[1].legend(
        facecolor='none',
        edgecolor='none',
        labelcolor='#64748b',
        framealpha=0.0,
        fontsize=8.5,
        loc='lower left',
    )
    annotate_terminal(axes[1], steps[-1], t_fulfill[-1], f"{t_fulfill[-1]:.1f}%", fulfillment_color)
    annotate_terminal(axes[1], steps[-1], r_fulfill[-1], f"{r_fulfill[-1]:.1f}%", random_color)

    fig1.tight_layout(pad=2.6)

    fig2, ax2 = plt.subplots(figsize=(16, 7), dpi=130)
    fig2.patch.set_facecolor('none')
    style_plot_axis(ax2)

    metrics      = ['Fulfillment %', 'Stockout %', 'Avg Reward x10', 'Total Reward']
    trained_vals = [
        t_summary.get('fulfillment_rate_%', 0),
        t_summary.get('stockout_rate_%', 0),
        t_summary.get('avg_reward', 0) * 10,
        t_total
    ]
    random_vals = [
        r_summary.get('fulfillment_rate_%', 0),
        r_summary.get('stockout_rate_%', 0),
        r_summary.get('avg_reward', 0) * 10,
        r_total
    ]

    x     = np.arange(len(metrics))
    width = 0.35
    bars1 = ax2.bar(x - width/2, trained_vals, width,
                    label='Trained agent', color=trained_color, alpha=0.9,
                    edgecolor='#c4b5fd', linewidth=1.1)
    bars2 = ax2.bar(x + width/2, random_vals,  width,
                    label='Random agent',  color=random_color, alpha=0.86,
                    edgecolor='#fda4af', linewidth=1.1)

    all_values = trained_vals + random_vals
    min_value = min(all_values)
    max_value = max(all_values)
    pad = max(8, (max_value - min_value) * 0.20)
    ax2.set_ylim(min_value - pad, max_value + pad)
    ax2.axhline(0, color='#e2e8f0', alpha=0.22, linewidth=1)
    ax2.set_title('Matched evaluation scoreboard', color='#64748b',
                  fontsize=14, fontweight='bold', loc='left')
    ax2.text(
        0.01, 0.95,
        f'{n_warehouses}W / {n_retailers}R / {len(seeds)} synchronized rollouts',
        transform=ax2.transAxes, color='#94a3b8', fontsize=8.5
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, color='#64748b', fontsize=9)
    ax2.set_ylabel('Score snapshot', color='#64748b')
    ax2.legend(
        facecolor='none',
        edgecolor='none',
        labelcolor='#64748b',
        framealpha=0.0,
        fontsize=8.5,
        loc='upper left',
    )
    label_bars(ax2, bars1)
    label_bars(ax2, bars2)

    fig2.tight_layout(pad=2.4)

    def winner(t, r, higher_better=True):
        if higher_better:
            return "Trained wins" if t > r else "Random wins"
        return "Trained wins" if t < r else "Random wins"

    t_fulfill_rate = t_summary.get('fulfillment_rate_%', 0)
    r_fulfill_rate = r_summary.get('fulfillment_rate_%', 0)
    t_stockout     = t_summary.get('stockout_rate_%', 0)
    r_stockout     = r_summary.get('stockout_rate_%', 0)
    t_avg          = t_summary.get('avg_reward', 0)
    r_avg          = r_summary.get('avg_reward', 0)

    improvement = ((t_total - r_total) / (abs(r_total) + 1e-9)) * 100
    highlights_html = build_highlights_html(
        n_warehouses=n_warehouses,
        n_retailers=n_retailers,
        model_name=model_name,
        algorithm_name=algorithm_name,
        trained_agent_loaded=trained_agent is not None,
        t_summary=t_summary,
        r_summary=r_summary,
        t_total=t_total,
        r_total=r_total,
        improvement=improvement,
        num_seeds=len(seeds),
    )

    summary_text = f"""
{'='*55}
  COMPARISON SUMMARY
  Config: {n_warehouses} Warehouses x {n_retailers} Retailers
  Average over {len(seeds)} matched evaluation episodes
{'='*55}

METRIC                TRAINED      RANDOM        RESULT
{'-'*55}
Fulfillment Rate   {t_fulfill_rate:>8.1f}%   {r_fulfill_rate:>8.1f}%    {winner(t_fulfill_rate, r_fulfill_rate)}
Stockout Rate      {t_stockout:>8.1f}%   {r_stockout:>8.1f}%    {winner(t_stockout, r_stockout, False)}
Avg Reward         {t_avg:>8.3f}    {r_avg:>8.3f}    {winner(t_avg, r_avg)}
Total Reward       {t_total:>8.3f}    {r_total:>8.3f}    {winner(t_total, r_total)}
Total Steps        {t_summary.get('total_steps', 0):>8}    {r_summary.get('total_steps', 0):>8}
{'-'*55}
Total Demand       {t_summary.get('total_demand', 0):>8.1f}    {r_summary.get('total_demand', 0):>8.1f}
Total Fulfilled    {t_summary.get('total_fulfilled', 0):>8.1f}    {r_summary.get('total_fulfilled', 0):>8.1f}
{'='*55}
Improvement: {improvement:+.1f}% better total reward vs random
Model file: {model_name}.zip ({algorithm_name})
{'='*55}
"""
    return highlights_html, fig1, fig2, summary_text


# ── Gradio UI ──
with gr.Blocks(title="Supply Chain RL", fill_width=True) as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Group(elem_classes="control-card"):
                gr.HTML(
                    """
                    <div class="panel-kicker">Scenario composer</div>
                    <h3 class="panel-title">Re-shape the network and replay the same shocks.</h3>
                    <p class="panel-copy">
                      Drag the sliders to change the supply network footprint, then launch the synchronized evaluation to see how the trained policy behaves under pressure.
                    </p>
                    """
                )
                warehouses = gr.Slider(2, 6, value=3, step=1, label="Warehouses in the network")
                retailers  = gr.Slider(3, 10, value=5, step=1, label="Retailers served")
                btn = gr.Button("Launch matched simulation", variant="primary", size="lg", elem_id="run-deck-button")
                gr.HTML(
                    """
                    <div class="control-footer">
                      <div class="control-stat">
                        <span>Matched seeds</span>
                        <strong>11 / 22 / 33 / 44 / 55</strong>
                      </div>
                      <div class="control-stat">
                        <span>Episode depth</span>
                        <strong>100 steps in each synchronized rollout</strong>
                      </div>
                      <div class="control-stat">
                        <span>Comparison rule</span>
                        <strong>Same shocks, same delays, different policy behavior</strong>
                      </div>
                    </div>
                    """
                )
        with gr.Column(scale=4):
            gr.HTML(GUIDE_PANEL_HTML)

    highlights = gr.HTML(value=DEFAULT_HIGHLIGHTS)

    with gr.Group(elem_classes="plot-shell"):
        gr.HTML(
            """
            <div class="panel-kicker">Trajectory view</div>
            <h3 class="plot-title">How each policy behaves over time</h3>
            <p class="plot-copy">
              Watch reward and fulfillment move through the episode horizon to see whether performance is consistent or fragile.
            </p>
            """
        )
        chart1 = gr.Plot(label="Average step-by-step performance")

    with gr.Group(elem_classes="plot-shell"):
        gr.HTML(
            """
            <div class="panel-kicker">Scoreboard</div>
            <h3 class="plot-title">At-a-glance policy comparison</h3>
            <p class="plot-copy">
              This panel compresses the matchup into the four business-facing numbers that matter most.
            </p>
            """
        )
        chart2 = gr.Plot(label="Matched evaluation comparison")

    with gr.Group(elem_classes="summary-shell"):
        gr.HTML(
            """
            <div class="panel-kicker">Detailed transcript</div>
            <h3 class="summary-title">Comparison log</h3>
            <p class="summary-copy">
              Use the detailed readout when you want the exact averaged totals behind the headline verdict.
            </p>
            """
        )
        summary = gr.Textbox(
            label="Detailed evaluation log",
            lines=20,
            value="Run a simulation to populate the evaluation summary.",
            show_label=False,
            elem_classes="log-panel",
        )

    btn.click(
        run_demo,
        inputs=[warehouses, retailers],
        outputs=[highlights, chart1, chart2, summary]
    )

ENABLE_TRANSLATION_JS = """
() => {
    document.documentElement.removeAttribute('translate');
    document.body.removeAttribute('translate');
    document.querySelectorAll('.notranslate').forEach(el => {
        el.classList.remove('notranslate');
    });
}
"""

if __name__ == "__main__":
    demo.launch(css=APP_CSS, js=ENABLE_TRANSLATION_JS)
