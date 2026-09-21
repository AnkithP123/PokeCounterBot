import os
import io
import time
import base64
import glob
import logging
from aiohttp import web
import cv2
import numpy as np
from PIL import Image

from src.ocr import extract_cp_from_image, _load_image
from src.classifier import classify_pokemon_from_image

logger = logging.getLogger("PokeDashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "downloaded_test_images")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PokéCounter Live Evaluator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #090d16;
    --surface: #111827;
    --surface-hover: #1f293d;
    --card: #151d30;
    --border: #23304a;
    --accent: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.25);
    --pokemon-yellow: #facc15;
    --pokemon-red: #f43f5e;
    --success: #10b981;
    --success-bg: rgba(16, 185, 129, 0.15);
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --radius: 16px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }

  /* Header */
  header {
    background: rgba(17, 24, 39, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .pokeball-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(180deg, #ef4444 50%, #ffffff 50%);
    border-radius: 50%;
    border: 3px solid #0f172a;
    position: relative;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.4);
  }
  .pokeball-icon::after {
    content: '';
    position: absolute;
    width: 10px;
    height: 10px;
    background: #fff;
    border: 3px solid #0f172a;
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }
  .brand h1 {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #fff 40%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .brand span {
    font-size: 11px;
    text-transform: uppercase;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    letter-spacing: 0.8px;
  }
  .header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-muted);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--success);
  }

  /* Main Container */
  main {
    flex: 1;
    max-width: 1380px;
    width: 100%;
    margin: 0 auto;
    padding: 28px 24px 60px;
    display: flex;
    flex-direction: column;
    gap: 28px;
  }

  /* Hero Section / Quick Samples */
  .quick-samples-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    overflow-x: auto;
  }
  .samples-label {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }
  .samples-list {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    scrollbar-width: thin;
  }
  .sample-chip {
    padding: 6px 14px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sample-chip:hover {
    background: var(--surface-hover);
    border-color: var(--accent);
    color: var(--accent);
    transform: translateY(-1px);
  }

  /* Workspace Layout */
  .evaluator-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 960px) {
    .evaluator-grid { grid-template-columns: 1fr; }
  }

  /* Dropzone Card */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .dropzone {
    border: 2px dashed var(--border);
    background: rgba(21, 29, 48, 0.5);
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: var(--accent-glow);
    box-shadow: 0 0 24px var(--accent-glow);
  }
  .dropzone input[type="file"] {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    opacity: 0;
    cursor: pointer;
  }
  .drop-icon {
    font-size: 42px;
    margin-bottom: 12px;
    display: inline-block;
  }
  .drop-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .drop-hint {
    font-size: 13px;
    color: var(--text-muted);
  }
  .paste-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 14px;
    padding: 6px 14px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 999px;
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
  }
  .kbd {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  /* Image Preview */
  .preview-wrapper {
    position: relative;
    max-height: 440px;
    border-radius: 12px;
    overflow: hidden;
    background: #000;
    display: none;
    justify-content: center;
    align-items: center;
  }
  .preview-img {
    max-width: 100%;
    max-height: 440px;
    object-fit: contain;
    border-radius: 8px;
  }
  .clear-btn {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border);
    color: #fff;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    cursor: pointer;
    backdrop-filter: blur(8px);
  }
  .clear-btn:hover { background: var(--pokemon-red); border-color: var(--pokemon-red); }

  /* Results Panel */
  .results-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 22px;
    min-height: 460px;
  }
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: var(--text-muted);
    padding: 40px;
  }
  .empty-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.6; }
  .empty-state h3 { font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 6px; }
  .empty-state p { font-size: 13px; max-width: 320px; }

  /* Evaluated Card Header */
  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }
  .species-title-group h2 {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #fff;
  }
  .species-subtitle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
  }
  .latency-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }
  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.5px;
  }
  .metric-value {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  .cp-val { color: var(--accent); }
  .hp-val { color: var(--success); }
  .candy-val { color: var(--pokemon-yellow); font-size: 18px; text-transform: uppercase; }

  /* Appraisal IV Bars */
  .iv-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .iv-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .iv-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
  .iv-total { font-size: 16px; font-weight: 800; color: var(--pokemon-yellow); }
  .iv-bars { display: flex; flex-direction: column; gap: 8px; }
  .iv-bar-row {
    display: grid;
    grid-template-columns: 64px 1fr 32px;
    align-items: center;
    gap: 12px;
    font-size: 12px;
  }
  .iv-track {
    height: 10px;
    background: #090d16;
    border-radius: 999px;
    overflow: hidden;
    position: relative;
    border: 1px solid var(--border);
  }
  .iv-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Explanation / Method */
  .method-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.5;
    color: #cbd5e1;
  }
  .method-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
    letter-spacing: 0.5px;
  }

  /* Session History Tray */
  .history-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
  }
  .history-title {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 14px;
  }
  .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
  }
  .history-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .history-card:hover {
    background: var(--surface-hover);
    border-color: var(--accent);
  }
  .hist-name { font-weight: 700; font-size: 14px; }
  .hist-meta { font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

  /* Loading Spinner */
  .spinner-overlay {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(4px);
    display: none;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    gap: 12px;
    z-index: 10;
  }
  .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="pokeball-icon"></div>
    <div>
      <h1>PokéCounter Live</h1>
    </div>
    <span>Sub-Second OCR Engine</span>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    <span>Engine Online (Tesseract 5 C++ Fast Path)</span>
  </div>
</header>

<main>
  <!-- Quick Test Samples Carousel -->
  <div class="quick-samples-bar">
    <span class="samples-label">⚡ Quick Test Samples:</span>
    <div class="samples-list" id="samplesContainer">
      <span style="font-size: 12px; color: var(--text-muted);">Loading sample screenshots...</span>
    </div>
  </div>

  <!-- Main Evaluator Grid -->
  <div class="evaluator-grid">
    <!-- Dropzone / Input Panel -->
    <div class="panel" style="position: relative;">
      <div class="spinner-overlay" id="spinner">
        <div class="spinner"></div>
        <div style="font-size: 14px; font-weight: 600;">Evaluating Screenshot...</div>
      </div>

      <div class="dropzone" id="dropzone">
        <input type="file" id="fileInput" accept="image/*">
        <div class="drop-icon">📸</div>
        <div class="drop-title">Drop or Paste Screenshot Here</div>
        <div class="drop-hint">Supports full screens, appraisals, cropped views, and ultra-wide layouts</div>
        <div class="paste-badge">
          <span>Clipboard Shortcut:</span>
          <span class="kbd">Cmd + V</span>
          <span>or</span>
          <span class="kbd">Ctrl + V</span>
        </div>
      </div>

      <div class="preview-wrapper" id="previewWrapper">
        <img src="" id="previewImg" class="preview-img" alt="Uploaded screenshot preview">
        <button class="clear-btn" id="clearBtn">✕ Clear Image</button>
      </div>
    </div>

    <!-- Results Panel -->
    <div class="results-panel" id="resultsPanel">
      <div class="empty-state" id="emptyState">
        <div class="empty-icon">⚡</div>
        <h3>No Screenshot Evaluated Yet</h3>
        <p>Paste an image from your clipboard (<span class="kbd">Cmd+V</span>), drop a file, or click any sample chip above to test sub-second detection.</p>
      </div>

      <div id="resultsContent" style="display: none; flex-direction: column; gap: 20px;">
        <div class="results-header">
          <div class="species-title-group">
            <h2 id="resSpecies">Species Name</h2>
            <div class="species-subtitle">
              <span>Evolution Family: <strong id="resFamily" style="color: var(--pokemon-yellow);">Family</strong></span>
              <span>•</span>
              <span id="resCandidates">1 Candidate</span>
            </div>
          </div>
          <div class="latency-tag" id="resLatency">⚡ 150 ms</div>
        </div>

        <div class="stats-grid">
          <div class="metric-card">
            <span class="metric-label">Combat Power (CP)</span>
            <span class="metric-value cp-val" id="resCp">---</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Hit Points (HP)</span>
            <span class="metric-value hp-val" id="resHp">---</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Evolution Family</span>
            <span class="metric-value candy-val" id="resFamilyBadge">---</span>
          </div>
        </div>

        <div class="stats-grid" id="powerupDetailsRow" style="margin-top: 14px;">
          <div class="metric-card">
            <span class="metric-label">⭐ Stardust Balance</span>
            <span class="metric-value" id="resStardust" style="color: #60a5fa; font-size: 20px;">---</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">🍬 Candy Stock</span>
            <span class="metric-value" id="resCandyStock" style="color: var(--pokemon-yellow); font-size: 20px;">---</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">⚡ Power Up & Level</span>
            <span class="metric-value" id="resPowerUp" style="color: #f43f5e; font-size: 17px; font-weight: 700;">---</span>
          </div>
        </div>

        <!-- IV Appraisal Bars (shown only if appraisal screen) -->
        <div class="iv-section" id="ivSection" style="display: none;">
          <div class="iv-header">
            <span class="iv-title">Appraisal IV Breakdown</span>
            <span class="iv-total" id="ivPercent">100% (45/45)</span>
          </div>
          <div class="iv-bars">
            <div class="iv-bar-row">
              <span>Attack</span>
              <div class="iv-track"><div class="iv-fill" id="atkFill" style="width: 100%;"></div></div>
              <strong id="atkVal">15</strong>
            </div>
            <div class="iv-bar-row">
              <span>Defense</span>
              <div class="iv-track"><div class="iv-fill" id="defFill" style="width: 100%;"></div></div>
              <strong id="defVal">15</strong>
            </div>
            <div class="iv-bar-row">
              <span>HP / Sta</span>
              <div class="iv-track"><div class="iv-fill" id="staFill" style="width: 100%;"></div></div>
              <strong id="staVal">15</strong>
            </div>
          </div>
        </div>

        <div class="method-box">
          <div class="method-title">Identification Logic & Proof</div>
          <div id="resExplanation">Explanation text...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Recent Session Evaluations -->
  <div class="history-section" id="historySection" style="display: none;">
    <div class="history-title">Session Evaluations</div>
    <div class="history-grid" id="historyGrid"></div>
  </div>
</main>

<script>
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewWrapper = document.getElementById('previewWrapper');
  const previewImg = document.getElementById('previewImg');
  const clearBtn = document.getElementById('clearBtn');
  const spinner = document.getElementById('spinner');

  const emptyState = document.getElementById('emptyState');
  const resultsContent = document.getElementById('resultsContent');
  const resSpecies = document.getElementById('resSpecies');
  const resFamily = document.getElementById('resFamily');
  const resCandidates = document.getElementById('resCandidates');
  const resLatency = document.getElementById('resLatency');
  const resCp = document.getElementById('resCp');
  const resHp = document.getElementById('resHp');
  const resFamilyBadge = document.getElementById('resFamilyBadge');
  const resStardust = document.getElementById('resStardust');
  const resCandyStock = document.getElementById('resCandyStock');
  const resPowerUp = document.getElementById('resPowerUp');
  const resExplanation = document.getElementById('resExplanation');

  const ivSection = document.getElementById('ivSection');
  const ivPercent = document.getElementById('ivPercent');
  const atkFill = document.getElementById('atkFill');
  const defFill = document.getElementById('defFill');
  const staFill = document.getElementById('staFill');
  const atkVal = document.getElementById('atkVal');
  const defVal = document.getElementById('defVal');
  const staVal = document.getElementById('staVal');

  const samplesContainer = document.getElementById('samplesContainer');
  const historySection = document.getElementById('historySection');
  const historyGrid = document.getElementById('historyGrid');

  const evaluationHistory = [];

  // Drag & drop handlers
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
  });
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Direct Clipboard Paste Listener (Cmd+V / Ctrl+V anywhere on screen)
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const blob = item.getAsFile();
        handleFile(blob);
        break;
      }
    }
  });

  clearBtn.addEventListener('click', () => {
    previewWrapper.style.display = 'none';
    dropzone.style.display = 'block';
    emptyState.style.display = 'flex';
    resultsContent.style.display = 'none';
    fileInput.value = '';
  });

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      previewImg.src = dataUrl;
      previewWrapper.style.display = 'flex';
      dropzone.style.display = 'none';
      evaluateImage(dataUrl);
    };
    reader.readAsDataURL(file);
  }

  async function evaluateImage(dataUrl) {
    spinner.style.display = 'flex';
    const tStart = performance.now();
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await res.json();
      renderResults(data, performance.now() - tStart);
    } catch (err) {
      alert('Evaluation error: ' + err.message);
    } finally {
      spinner.style.display = 'none';
    }
  }

  function renderResults(data, clientTime) {
    emptyState.style.display = 'none';
    resultsContent.style.display = 'flex';

    resSpecies.textContent = data.species || 'Unknown Pokémon';
    resFamily.textContent = data.candy_family || 'Unknown Family';
    resCandidates.textContent = data.candidates && data.candidates.length > 0 
      ? `${data.candidates.length} Candidate${data.candidates.length > 1 ? 's' : ''}` 
      : 'Direct Match';

    const serverMs = data.latency_ms ? Math.round(data.latency_ms) : Math.round(clientTime);
    resLatency.innerHTML = `⚡ ${serverMs} ms`;
    resLatency.style.color = serverMs < 400 ? 'var(--success)' : (serverMs < 800 ? 'var(--pokemon-yellow)' : 'var(--pokemon-red)');

    resCp.textContent = data.cp !== null ? `CP ${data.cp}` : 'Unknown CP';
    resHp.textContent = data.hp !== null ? `${data.hp} HP` : 'Unknown HP';
    resFamilyBadge.textContent = data.candy_family ? data.candy_family : 'Unknown Family';

    resStardust.textContent = (data.stardust !== null && data.stardust !== undefined) ? data.stardust.toLocaleString() : 'Not detected';
    resCandyStock.textContent = (data.candy_count !== null && data.candy_count !== undefined) ? `${data.candy_count}` : 'Not detected';

    if (data.powerup_stardust) {
      const puCandy = data.powerup_candy ? ` + ${data.powerup_candy} 🍬` : '';
      const lvl = data.estimated_level ? ` (${data.estimated_level})` : '';
      resPowerUp.textContent = `${data.powerup_stardust.toLocaleString()} ⭐${puCandy}${lvl}`;
    } else {
      resPowerUp.textContent = 'Not detected';
    }

    resExplanation.textContent = data.explanation || 'Could not determine species or stats from this screenshot.';

    if (data.appraisal_ivs) {
      ivSection.style.display = 'flex';
      const ivs = data.appraisal_ivs;
      ivPercent.textContent = `${ivs.percent}% (${ivs.atk + ivs.def + ivs.sta}/45)`;
      atkVal.textContent = ivs.atk;
      defVal.textContent = ivs.def;
      staVal.textContent = ivs.sta;
      atkFill.style.width = `${(ivs.atk / 15) * 100}%`;
      defFill.style.width = `${(ivs.def / 15) * 100}%`;
      staFill.style.width = `${(ivs.sta / 15) * 100}%`;
    } else {
      ivSection.style.display = 'none';
    }

    // Add to session history
    addToHistory(data, serverMs);
  }

  function addToHistory(data, ms) {
    historySection.style.display = 'block';
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
      <div>
        <div class="hist-name">${data.species || 'Unknown Pokémon'}</div>
        <div class="hist-meta">${data.cp ? 'CP ' + data.cp : 'Unknown CP'}${data.hp ? ' • ' + data.hp + ' HP' : ''}</div>
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: ${ms < 400 ? 'var(--success)' : 'var(--pokemon-yellow)'};">
        ${ms}ms
      </div>
    `;
    historyGrid.prepend(card);
  }

  // Load sample screenshots from backend
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const samples = await res.json();
      samplesContainer.innerHTML = '';
      samples.forEach(s => {
        const chip = document.createElement('div');
        chip.className = 'sample-chip';
        chip.textContent = s.label;
        chip.onclick = () => loadSampleImage(s.filename);
        samplesContainer.appendChild(chip);
      });
    } catch (e) {
      samplesContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">Could not load samples</span>';
    }
  }

  async function loadSampleImage(filename) {
    spinner.style.display = 'flex';
    try {
      const res = await fetch(`/api/sample/${filename}`);
      const blob = await res.blob();
      handleFile(blob);
    } catch (e) {
      alert('Failed to load sample image: ' + e);
      spinner.style.display = 'none';
    }
  }

  loadSamples();
</script>
</body>
</html>
"""


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML_CONTENT, content_type="text/html")


async def evaluate_handler(request: web.Request) -> web.Response:
    t0 = time.perf_counter()

    image_bytes = None
    if request.content_type.startswith("multipart/form-data"):
        reader = await request.multipart()
        field = await reader.next()
        if field and field.name == "image":
            image_bytes = await field.read()
    else:
        try:
            data = await request.json()
            raw_img = data.get("image") or data.get("image_base64")
            if raw_img:
                if "," in raw_img:
                    raw_img = raw_img.split(",", 1)[1]
                image_bytes = base64.b64decode(raw_img)
        except Exception as e:
            logger.warning("Error parsing JSON body: %s", e)

    if not image_bytes:
        return web.json_response({"success": False, "error": "No image data provided."}, status=400)

    try:
        # Load image via our robust _load_image
        img = _load_image(image_bytes)
        try:
            with open("/tmp/last_evaluate_image.png", "wb") as f_dbg:
                f_dbg.write(image_bytes)
        except Exception:
            pass

        # 1. Extract CP
        t_cp_0 = time.perf_counter()
        cp = extract_cp_from_image(img)
        cp_ms = (time.perf_counter() - t_cp_0) * 1000.0

        # 2. Extract Species / Stats with known_cp
        t_sp_0 = time.perf_counter()
        res = classify_pokemon_from_image(img, known_cp=cp)
        sp_ms = (time.perf_counter() - t_sp_0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        validated_cp = res.get("cp") or cp
        logger.info("Evaluated: species=%s, cp=%s, hp=%s, stardust=%s, candy=%s, powerup_dust=%s in %.1f ms",
                    res.get("species"), validated_cp, res.get("hp"), res.get("stardust"), res.get("candy_count"), res.get("powerup_stardust"), total_ms)

        return web.json_response({
            "success": True,
            "species": res.get("species"),
            "cp": validated_cp,
            "hp": res.get("hp"),
            "candy_family": res.get("candy_family"),
            "candidates": res.get("candidates", []),
            "appraisal_ivs": res.get("appraisal_ivs"),
            "stardust": res.get("stardust"),
            "candy_count": res.get("candy_count"),
            "powerup_stardust": res.get("powerup_stardust"),
            "powerup_candy": res.get("powerup_candy"),
            "estimated_level": res.get("estimated_level"),
            "explanation": res.get("explanation"),
            "latency_ms": round(total_ms, 1),
            "timing": {
                "cp_ms": round(cp_ms, 1),
                "species_ms": round(sp_ms, 1),
                "total_ms": round(total_ms, 1)
            }
        })
    except Exception as ex:
        logger.exception("Evaluation failed: %s", ex)
        return web.json_response({"success": False, "error": str(ex)}, status=500)


async def samples_handler(request: web.Request) -> web.Response:
    samples = []
    if os.path.exists(SAMPLES_DIR):
        files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.*")))
        valid_exts = {".png", ".jpg", ".jpeg", ".heic"}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                fname = os.path.basename(f)
                parts = fname.split("_")
                label = f"Sample #{parts[0]}" if parts[0].isdigit() else fname[:16]
                samples.append({"filename": fname, "label": label})
    return web.json_response(samples[:15])


async def sample_file_handler(request: web.Request) -> web.StreamResponse:
    fname = request.match_info.get("filename")
    if not fname:
        return web.Response(status=404)
    safe_name = os.path.basename(fname)
    fpath = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(fpath):
        return web.Response(status=404, text="File not found")
    return web.FileResponse(fpath)


def create_app() -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_get("/", index_handler)
    app.router.add_post("/api/evaluate", evaluate_handler)
    app.router.add_get("/api/samples", samples_handler)
    app.router.add_get("/api/sample/{filename}", sample_file_handler)
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PokéCounter Live Web Dashboard")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind (default: 8090)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\\n========================================================")
    print(f"🚀 PokéCounter Live Evaluator running at:")
    print(f"👉 http://{args.host}:{args.port}")
    print(f"========================================================\\n")
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
