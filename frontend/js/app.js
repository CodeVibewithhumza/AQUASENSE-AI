/**
 * 💧 AquaSense AI — Single Page Web Application Engine
 * High-performance, reactive, and explainable water quality diagnostics.
 */

const API_BASE = window.location.origin;

// State Management
let datasetSummary = null;
let charts = {};

// WHO Guidelines Reference Ranges
const WHO_GUIDELINES = {
  ph: { min: 6.5, max: 8.5, format: v => parseFloat(v).toFixed(2) },
  Hardness: { min: 100, max: 300, format: v => parseFloat(v).toFixed(1) },
  Solids: { min: 0, max: 25000, format: v => Number(Math.round(v)).toLocaleString() },
  Chloramines: { min: 4.0, max: 10.0, format: v => parseFloat(v).toFixed(2) },
  Sulfate: { min: 250, max: 400, format: v => parseFloat(v).toFixed(1) },
  Conductivity: { min: 200, max: 600, format: v => parseFloat(v).toFixed(1) },
  Organic_carbon: { min: 5.0, max: 20.0, format: v => parseFloat(v).toFixed(2) },
  Trihalomethanes: { min: 20.0, max: 80.0, format: v => parseFloat(v).toFixed(1) },
  Turbidity: { min: 1.0, max: 5.0, format: v => parseFloat(v).toFixed(2) }
};

// DOM Initializer
document.addEventListener('DOMContentLoaded', () => {
  initMobileMenu();
  initTabs();
  initDateDisplay();
  initSliders();
  initPresets();
  loadDataExplorer();
  loadModelCompare();
  loadExplainability();
  loadHistory();

  // Button Bindings
  const analyzeBtn = document.getElementById('btn-analyze');
  if (analyzeBtn) analyzeBtn.addEventListener('click', handlePrediction);
  
  const refreshBtn = document.getElementById('btn-refresh-history');
  if (refreshBtn) refreshBtn.addEventListener('click', loadHistory);

  const featureSelect = document.getElementById('feature-dist-select');
  if (featureSelect) {
    featureSelect.addEventListener('change', (e) => {
      renderDistributionChart(e.target.value);
    });
  }
});

/**
 * Responsive Mobile Drawer & Hamburger Menu Controller
 */
function initMobileMenu() {
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  const closeBtn = document.getElementById('sidebar-close-btn');

  function openSidebar() {
    if (sidebar) sidebar.classList.add('open');
    if (overlay) overlay.classList.add('active');
    if (hamburgerBtn) hamburgerBtn.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeSidebar() {
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('active');
    if (hamburgerBtn) hamburgerBtn.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (hamburgerBtn) {
    hamburgerBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (sidebar && sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      closeSidebar();
    });
  }

  if (overlay) {
    overlay.addEventListener('click', closeSidebar);
  }

  // Close with Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar && sidebar.classList.contains('open')) {
      closeSidebar();
    }
  });

  // Attach close handler to tab buttons so switching tabs automatically closes drawer
  const tabBtns = document.querySelectorAll('.sidebar-nav .nav-item');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (window.innerWidth <= 1024) {
        closeSidebar();
      }
    });
  });
}

/**
 * Live Date Formatter
 */
function initDateDisplay() {
  const dateEl = document.getElementById('live-date-str');
  if (dateEl) {
    const now = new Date();
    const options = { month: 'short', day: 'numeric', year: 'numeric', weekday: 'long' };
    const formatted = now.toLocaleDateString('en-US', options);
    // e.g. "Sep 10, 2026 • Thursday"
    const parts = formatted.split(', ');
    if (parts.length >= 3) {
      dateEl.innerHTML = `${parts[1]}, ${parts[2]} &bull; ${parts[0]}`;
    } else {
      dateEl.textContent = formatted;
    }
  }
}

/**
 * Tab Navigation Routing
 */
function initTabs() {
  const tabBtns = document.querySelectorAll('.sidebar-nav .nav-item');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      switchTab(targetId);
    });
  });
}

function switchTab(targetId) {
  const tabBtns = document.querySelectorAll('.sidebar-nav .nav-item');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(b => {
    if (b.getAttribute('data-tab') === targetId) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });

  tabPanels.forEach(panel => {
    if (panel.id === targetId) {
      panel.classList.add('active');
    } else {
      panel.classList.remove('active');
    }
  });

  // Auto-close mobile drawer if open
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  const hamburgerBtn = document.getElementById('hamburger-btn');
  if (sidebar) sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('active');
  if (hamburgerBtn) hamburgerBtn.classList.remove('active');
  document.body.style.overflow = '';

  // Lazy refresh for charts when tab is activated
  if (targetId === 'tab-dashboard' || targetId === 'tab-explorer') loadDataExplorer();
  if (targetId === 'tab-compare') loadModelCompare();
  if (targetId === 'tab-explain') loadExplainability();
  if (targetId === 'tab-history') loadHistory();
}

/**
 * Slider Synchronizer & WHO Compliance Status Checkers
 */
function initSliders() {
  const sliderInputs = document.querySelectorAll('.range-input');
  sliderInputs.forEach(slider => {
    const key = slider.id.replace('in_', '');
    updateSliderVisuals(slider, key);

    slider.addEventListener('input', () => {
      updateSliderVisuals(slider, key);
    });
  });
}

function updateSliderVisuals(slider, key) {
  const val = parseFloat(slider.value);
  const badge = document.getElementById(`${slider.id}-val`);
  const statusDot = document.getElementById(`${slider.id}-status`);
  const guideline = WHO_GUIDELINES[key];

  if (badge && guideline) {
    badge.textContent = guideline.format(val);
  }

  if (statusDot && guideline) {
    const isCompliant = val >= guideline.min && val <= guideline.max;
    if (isCompliant) {
      statusDot.className = 'status-dot safe';
      statusDot.title = 'In WHO Safe Guideline Range';
    } else {
      statusDot.className = 'status-dot danger';
      statusDot.title = `Exceeds WHO Safe Limits (${guideline.min} - ${guideline.max})`;
    }
  }
}

/**
 * Clinical Baseline Preset Handlers
 */
function initPresets() {
  const presets = {
    who: {
      ph: 7.2, Hardness: 160.0, Solids: 18000.0, Chloramines: 7.1,
      Sulfate: 330.0, Conductivity: 410.0, Organic_carbon: 13.5,
      Trihalomethanes: 60.0, Turbidity: 3.2
    },
    mineral: {
      ph: 8.8, Hardness: 280.0, Solids: 38000.0, Chloramines: 4.5,
      Sulfate: 420.0, Conductivity: 650.0, Organic_carbon: 18.2,
      Trihalomethanes: 95.0, Turbidity: 5.8
    },
    contaminated: {
      ph: 4.2, Hardness: 85.0, Solids: 46000.0, Chloramines: 11.5,
      Sulfate: 180.0, Conductivity: 720.0, Organic_carbon: 24.0,
      Trihalomethanes: 110.0, Turbidity: 6.4
    }
  };

  document.querySelectorAll('.preset-pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const pName = btn.getAttribute('data-preset');
      const vals = presets[pName];
      if (vals) {
        applyPresetValues(vals);
      }
    });
  });
}

function applyPresetValues(vals) {
  for (const [key, val] of Object.entries(vals)) {
    const slider = document.getElementById(`in_${key}`);
    if (slider) {
      slider.value = val;
      updateSliderVisuals(slider, key);
    }
  }
}

/**
 * TAB 2: Diagnostic Prediction & Local Shapley XAI
 */
async function handlePrediction() {
  const btn = document.getElementById('btn-analyze');
  const spinner = document.getElementById('predict-spinner');
  const emptyState = document.getElementById('predict-empty-state');
  const resultCard = document.getElementById('predict-result-card');

  btn.disabled = true;
  emptyState.style.display = 'none';
  spinner.style.display = 'block';
  resultCard.style.display = 'none';

  const payload = {
    ph: parseFloat(document.getElementById('in_ph').value),
    Hardness: parseFloat(document.getElementById('in_Hardness').value),
    Solids: parseFloat(document.getElementById('in_Solids').value),
    Chloramines: parseFloat(document.getElementById('in_Chloramines').value),
    Sulfate: parseFloat(document.getElementById('in_Sulfate').value),
    Conductivity: parseFloat(document.getElementById('in_Conductivity').value),
    Organic_carbon: parseFloat(document.getElementById('in_Organic_carbon').value),
    Trihalomethanes: parseFloat(document.getElementById('in_Trihalomethanes').value),
    Turbidity: parseFloat(document.getElementById('in_Turbidity').value),
    model: document.getElementById('model-select').value
  };

  try {
    // 1. Run Probability & Potability Inference
    const pRes = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const predData = await pRes.json();

    // 2. Run Local SHAP Attribution
    const expRes = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const expData = await expRes.json();

    renderPredictionResult(payload, predData, expData);
    
    // Automatically update audit log in the background
    loadHistory();

  } catch (e) {
    console.error("Diagnostic inference failed:", e);
    alert("Inference failed. Please check backend logs.");
    emptyState.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    resultCard.style.display = 'block';
  }
}

function renderPredictionResult(inputSample, predData, expData) {
  const isPotable = predData.prediction === 1;
  const resultBox = document.getElementById('result-box');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictDesc = document.getElementById('verdict-desc');
  const confPill = document.getElementById('confidence-pill');

  if (isPotable) {
    resultBox.className = 'result-hero-box potable';
    verdictIcon.textContent = '💧';
    verdictTitle.textContent = 'POTABLE WATER';
    verdictDesc.textContent = 'Safe for Direct Human Consumption according to WHO guidelines';
    confPill.className = 'badge-pill badge-potable';
    confPill.textContent = `Confidence: ${predData.confidence.toUpperCase()}`;
  } else {
    resultBox.className = 'result-hero-box not-potable';
    verdictIcon.textContent = '⚠️';
    verdictTitle.textContent = 'NON-POTABLE WATER';
    verdictDesc.textContent = 'Unsafe for Direct Drinking — Purification Treatment Required';
    confPill.className = 'badge-pill badge-danger';
    confPill.textContent = `Confidence: ${predData.confidence.toUpperCase()}`;
  }

  // Probability Meter
  const pct = (predData.probability_potable * 100).toFixed(1);
  document.getElementById('prob-number').textContent = `${pct}%`;
  const bar = document.getElementById('prob-bar-fill');
  bar.style.width = `${pct}%`;
  bar.style.background = isPotable ? 'var(--safe-green)' : 'var(--danger-red)';

  // Render SHAP Factor Cards
  const posContainer = document.getElementById('factors-for-container');
  const negContainer = document.getElementById('factors-against-container');
  posContainer.innerHTML = '';
  negContainer.innerHTML = '';

  const shapVals = expData.shap_values || {};
  const topFor = expData.top_factors_for_potable || [];
  const topAgainst = expData.top_factors_against_potable || [];

  if (topFor.length === 0) {
    posContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.82rem; padding: 4px;">No strong positive drivers.</div>';
  } else {
    topFor.slice(0, 3).forEach(feat => {
      const val = shapVals[feat] || 0.0;
      const curVal = inputSample[feat] !== undefined ? inputSample[feat] : '-';
      posContainer.innerHTML += `
        <div style="background: var(--safe-green-light); border: 1px solid var(--safe-green-border); border-radius: var(--radius-sm); padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;">
          <div>
            <strong style="color: var(--navy-dark);">${feat}</strong>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">Val: ${curVal}</div>
          </div>
          <span style="color: var(--safe-green); font-weight: 800; font-family: 'JetBrains Mono';">+${val.toFixed(3)}</span>
        </div>`;
    });
  }

  if (topAgainst.length === 0) {
    negContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.82rem; padding: 4px;">No strong negative drivers.</div>';
  } else {
    topAgainst.slice(0, 3).forEach(feat => {
      const val = shapVals[feat] || 0.0;
      const curVal = inputSample[feat] !== undefined ? inputSample[feat] : '-';
      negContainer.innerHTML += `
        <div style="background: var(--danger-red-light); border: 1px solid var(--danger-red-border); border-radius: var(--radius-sm); padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;">
          <div>
            <strong style="color: var(--navy-dark);">${feat}</strong>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">Val: ${curVal}</div>
          </div>
          <span style="color: var(--danger-red); font-weight: 800; font-family: 'JetBrains Mono';">${val.toFixed(3)}</span>
        </div>`;
    });
  }

  // Generate Narrative
  const narrativeEl = document.getElementById('clinical-narrative-text');
  let narrative = '';
  if (isPotable) {
    narrative = `The calibrated <strong>${predData.model_used}</strong> model classified this water sample as <strong>POTABLE</strong> with <strong>${pct}%</strong> confidence. `;
    if (topFor.length > 0) {
      narrative += `Key positive factors are optimal levels in <strong>${topFor.slice(0, 2).join('</strong> and <strong>')}</strong>, safely within WHO potable standards.`;
    }
  } else {
    narrative = `The calibrated <strong>${predData.model_used}</strong> model classified this water sample as <strong>NON-POTABLE</strong> with <strong>${(100 - pct).toFixed(1)}%</strong> risk probability. `;
    if (topAgainst.length > 0) {
      narrative += `The verdict is primarily driven by hazardous levels in <strong>${topAgainst.slice(0, 2).join('</strong> and <strong>')}</strong>. Water purification is strongly recommended.`;
    }
  }
  narrativeEl.innerHTML = narrative;
}

/**
 * TAB 1 & 3: Data Explorer & Dashboard Loader
 */
async function loadDataExplorer() {
  try {
    const res = await fetch(`${API_BASE}/data/summary`);
    if (!res.ok) return;
    datasetSummary = await res.json();

    const summary = datasetSummary.summary;
    const stats = datasetSummary.stats;

    // Dashboard & Explorer KPI Cards
    const totalRows = summary.rows.toLocaleString();
    const potCnt = summary.class_distribution["1"] || 1278;
    const nonPotCnt = summary.class_distribution["0"] || 1998;
    const potPct = `${summary.class_distribution_pct["1"]}%`;
    const nonPotPct = `${summary.class_distribution_pct["0"]}%`;

    // Update Dashboard KPIs
    const dTot = document.getElementById('dash-total-samples');
    if (dTot) dTot.textContent = totalRows;
    const dPot = document.getElementById('dash-potable-pct');
    if (dPot) dPot.textContent = potPct;
    const dNonPot = document.getElementById('dash-non-potable-pct');
    if (dNonPot) dNonPot.textContent = nonPotPct;
    const dFeat = document.getElementById('dash-features-count');
    if (dFeat) dFeat.textContent = '15';

    // Update Explorer KPIs
    const eTot = document.getElementById('kpi-total-rows');
    if (eTot) eTot.textContent = totalRows;
    const ePot = document.getElementById('kpi-potable');
    if (ePot) ePot.textContent = potPct;
    const eNonPot = document.getElementById('kpi-non-potable');
    if (eNonPot) eNonPot.textContent = nonPotPct;

    // Render Dashboard & Explorer Charts
    renderDashboardDonut(potCnt, nonPotCnt);
    renderDashboardMissingChart(summary.missing_counts);
    renderExplorerDonut(potCnt, nonPotCnt);
    renderExplorerMissingChart(summary.missing_counts);
    renderStatsTable(stats);
    renderDistributionChart('ph');

  } catch (e) {
    console.error("Failed to load dataset summary:", e);
  }
}

/**
 * Dashboard Donut Chart (Center Label: "3,276 Samples")
 */
function renderDashboardDonut(potable, nonPotable) {
  const canvas = document.getElementById('chart-dash-donut');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['dashDonut']) charts['dashDonut'].destroy();

  // Custom center text plugin
  const centerTextPlugin = {
    id: 'centerText',
    beforeDraw(chart) {
      const { width, height, ctx } = chart;
      ctx.restore();
      const fontSize = (height / 115).toFixed(2);
      ctx.font = `800 ${fontSize}em Outfit, sans-serif`;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#0D2137';

      const text = (potable + nonPotable).toLocaleString();
      const textX = Math.round((width - ctx.measureText(text).width) / 2);
      const textY = height / 2 - 8;
      ctx.fillText(text, textX, textY);

      ctx.font = `600 ${(height / 230).toFixed(2)}em Inter, sans-serif`;
      ctx.fillStyle = '#8CA0B4';
      const subText = 'Samples';
      const subX = Math.round((width - ctx.measureText(subText).width) / 2);
      ctx.fillText(subText, subX, textY + 20);
      ctx.save();
    }
  };

  charts['dashDonut'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Potable', 'Non-Potable'],
      datasets: [{
        data: [potable, nonPotable],
        backgroundColor: ['#007BFF', '#66B2FF'],
        borderColor: '#FFFFFF',
        borderWidth: 3,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(context) {
              const total = potable + nonPotable;
              const pct = ((context.raw / total) * 100).toFixed(1);
              return ` ${context.label}: ${context.raw.toLocaleString()} (${pct}%)`;
            }
          }
        }
      },
      cutout: '72%'
    },
    plugins: [centerTextPlugin]
  });
}

/**
 * Dashboard Feature Overview Missing Values Chart
 */
function renderDashboardMissingChart(missingCounts) {
  const canvas = document.getElementById('chart-dash-missing');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['dashMissing']) charts['dashMissing'].destroy();

  const labels = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'];
  const keyMap = {
    'pH': 'ph', 'Hardness': 'Hardness', 'Solids': 'Solids', 'Chloramines': 'Chloramines',
    'Sulfate': 'Sulfate', 'Conductivity': 'Conductivity', 'Organic Carbon': 'Organic_carbon',
    'Trihalomethanes': 'Trihalomethanes', 'Turbidity': 'Turbidity'
  };
  const values = labels.map(l => missingCounts[keyMap[l]] || 0);

  charts['dashMissing'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: '#66B2FF',
        borderRadius: 4,
        barPercentage: 0.6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#57708C', font: { family: 'Inter', size: 10 }, maxRotation: 45, minRotation: 35 }
        },
        y: {
          grid: { color: '#EBF2F8' },
          ticks: { color: '#8CA0B4', font: { family: 'JetBrains Mono', size: 10 }, stepSize: 200 },
          max: 600
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function renderExplorerDonut(potable, nonPotable) {
  const canvas = document.getElementById('chart-class-donut');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['classDonut']) charts['classDonut'].destroy();

  charts['classDonut'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Non-Potable (0)', 'Potable (1)'],
      datasets: [{
        data: [nonPotable, potable],
        backgroundColor: ['#EF4444', '#10B981'],
        borderColor: '#FFFFFF',
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#12283E', font: { family: 'Outfit', size: 12, weight: 600 } }
        }
      },
      cutout: '68%'
    }
  });
}

function renderExplorerMissingChart(missingCounts) {
  const canvas = document.getElementById('chart-missing-values');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['missingValues']) charts['missingValues'].destroy();

  const labels = Object.keys(missingCounts).filter(k => missingCounts[k] > 0);
  const values = labels.map(k => missingCounts[k]);

  charts['missingValues'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Missing Observations',
        data: values,
        backgroundColor: '#007BFF',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { grid: { color: '#EBF2F8' }, ticks: { color: '#8CA0B4' } },
        y: { grid: { display: false }, ticks: { color: '#12283E', font: { family: 'Outfit', weight: 600 } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function renderStatsTable(stats) {
  const tbody = document.getElementById('stats-table-body');
  if (!tbody || !stats) return;

  tbody.innerHTML = '';
  for (const [key, s] of Object.entries(stats)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color: var(--primary-blue); font-weight: 700;">${s.name} (${key})</td>
      <td style="color: var(--safe-green); font-weight: 600;">${s.who_min} – ${s.who_max} ${s.unit}</td>
      <td>${s.mean} ± ${s.std}</td>
      <td>${s.median}</td>
      <td style="color: var(--text-secondary);">${s.min} – ${s.max}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderDistributionChart(featureKey) {
  if (!datasetSummary || !datasetSummary.distributions || !datasetSummary.distributions[featureKey]) return;
  const dist = datasetSummary.distributions[featureKey];

  const canvas = document.getElementById('chart-feature-dist');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['featureDist']) charts['featureDist'].destroy();

  const potVals = dist.potable;
  const nonPotVals = dist.non_potable;

  charts['featureDist'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: 25 }, (_, i) => i + 1),
      datasets: [
        {
          label: 'Potable Samples Density',
          data: potVals.slice(0, 25),
          borderColor: '#10B981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 2
        },
        {
          label: 'Non-Potable Samples Density',
          data: nonPotVals.slice(0, 25),
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#EBF2F8' }, ticks: { color: '#8CA0B4' } },
        y: { grid: { color: '#EBF2F8' }, ticks: { color: '#8CA0B4' } }
      },
      plugins: {
        legend: { labels: { color: '#12283E', font: { family: 'Outfit', weight: 600 } } }
      }
    }
  });
}

/**
 * TAB 4: Model Arena Leaderboard Loader
 */
async function loadModelCompare() {
  try {
    const res = await fetch(`${API_BASE}/models?_t=${Date.now()}`);
    if (!res.ok) return;
    const data = await res.json();
    
    // Strict 5 active models filter
    const ALLOWED_MODELS = ['random_forest', 'xgboost', 'svm', 'decision_tree', 'logistic_regression'];
    const models = (data.models || []).filter(m => ALLOWED_MODELS.includes(m.name));

    const champTitle = document.getElementById('champion-title');
    if (champTitle && data.best_model) {
      champTitle.textContent = `CHAMPION: ${data.best_model.replace('_', ' ').toUpperCase()} CLASSIFIER`;
    }

    const tbody = document.getElementById('models-table-body');
    if (tbody) {
      tbody.innerHTML = '';
      const names = [];
      const accuracies = [];
      const f1s = [];
      const aucs = [];
      const mccs = [];

      models.forEach(m => {
        names.push(m.name.replace('_', ' '));
        accuracies.push(m.accuracy);
        f1s.push(m.f1);
        aucs.push(m.roc_auc);
        mccs.push(m.mcc);

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="color: ${m.is_best ? 'var(--primary-blue)' : 'var(--navy-dark)'}; font-weight: 700;">
            ${m.name.replace('_', ' ').toUpperCase()} ${m.is_best ? '🏆' : ''}
          </td>
          <td style="${m.is_best ? 'color: var(--safe-green); font-weight: 800;' : ''}">${m.accuracy.toFixed(4)}</td>
          <td>${m.precision.toFixed(4)}</td>
          <td>${m.recall.toFixed(4)}</td>
          <td>${m.f1.toFixed(4)}</td>
          <td>${m.roc_auc.toFixed(4)}</td>
          <td style="color: var(--primary-blue); font-weight: 700;">${m.mcc.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      });

      renderModelBarChart(names, accuracies, f1s, aucs, mccs);
      renderRadarChart(models.slice(0, 3));
    }

  } catch (e) {
    console.error("Failed to load model arena:", e);
  }
}

function renderModelBarChart(names, accuracies, f1s, aucs, mccs) {
  const canvas = document.getElementById('chart-models-bar');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['modelsBar']) charts['modelsBar'].destroy();

  charts['modelsBar'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        { label: 'Accuracy', data: accuracies, backgroundColor: '#007BFF', borderRadius: 4 },
        { label: 'F1 Score', data: f1s, backgroundColor: '#10B981', borderRadius: 4 },
        { label: 'ROC-AUC', data: aucs, backgroundColor: '#F59E0B', borderRadius: 4 },
        { label: 'MCC', data: mccs, backgroundColor: '#8B5CF6', borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { color: '#12283E', font: { family: 'Outfit', weight: 600 }, maxRotation: 25 } },
        y: { grid: { color: '#EBF2F8' }, ticks: { color: '#8CA0B4' }, min: 0, max: 1 }
      },
      plugins: {
        legend: { labels: { color: '#12283E', font: { family: 'Outfit', weight: 600 } } }
      }
    }
  });
}

function renderRadarChart(topModels) {
  const canvas = document.getElementById('chart-radar');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['radar']) charts['radar'].destroy();

  const palette = ['#007BFF', '#10B981', '#F59E0B'];
  const datasets = topModels.map((m, idx) => ({
    label: m.name.replace('_', ' ').toUpperCase(),
    data: [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc, Math.max(0, m.mcc)],
    borderColor: palette[idx % palette.length],
    backgroundColor: `${palette[idx % palette.length]}22`,
    fill: true,
    borderWidth: 2.5
  }));

  charts['radar'] = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'MCC'],
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: '#EBF2F8' },
          angleLines: { color: '#EBF2F8' },
          pointLabels: { color: '#12283E', font: { family: 'Outfit', size: 11, weight: 600 } },
          ticks: { color: '#8CA0B4', backdropColor: 'transparent' },
          min: 0, max: 1
        }
      },
      plugins: { legend: { labels: { color: '#12283E', font: { family: 'Outfit', weight: 600 } } } }
    }
  });
}

/**
 * TAB 5: Explainability (XAI) Loader
 */
async function loadExplainability() {
  try {
    const fiRes = await fetch(`${API_BASE}/feature-importance`);
    if (fiRes.ok) {
      const fiData = await fiRes.json();
      renderFeatureImportanceChart(fiData.rankings);
    }

    const cRes = await fetch(`${API_BASE}/explain/consistency`);
    if (cRes.ok) {
      const cData = await cRes.json();
      renderAgreementTable(cData.agreement_table);
      renderConsistencyCorrelation(cData.correlation_matrix);
      const repEl = document.getElementById('consistency-report-text');
      if (repEl) repEl.innerHTML = cData.report.replace(/\n/g, '<br>');
    }

  } catch (e) {
    console.error("Failed to load explainability data:", e);
  }
}

function renderFeatureImportanceChart(rankings) {
  const canvas = document.getElementById('chart-shap-importance');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (charts['shapBar']) charts['shapBar'].destroy();

  const top10 = rankings.slice(0, 10).reverse();
  const labels = top10.map(r => r.feature);
  const values = top10.map(r => r.importance);

  charts['shapBar'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean |SHAP Value|',
        data: values,
        backgroundColor: '#007BFF',
        borderRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { grid: { color: '#EBF2F8' }, ticks: { color: '#8CA0B4' } },
        y: { grid: { display: false }, ticks: { color: '#12283E', font: { family: 'Outfit', weight: 600 } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function renderAgreementTable(agreeData) {
  const tbody = document.getElementById('agreement-table-body');
  if (!tbody || !agreeData) return;

  tbody.innerHTML = '';
  agreeData.forEach(row => {
    let badgeClass = 'badge-strong';
    if (row.Agreement === 'Moderate') badgeClass = 'badge-moderate';
    if (row.Agreement === 'Weak') badgeClass = 'badge-weak';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color: var(--primary-blue); font-weight: 700;">${row.Feature}</td>
      <td>#${row.SHAP_Rank}</td>
      <td>#${row.LIME_Rank}</td>
      <td>#${row.Perm_Rank}</td>
      <td>${row.Max_Rank_Diff}</td>
      <td><span class="badge-pill ${badgeClass}">${row.Agreement}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConsistencyCorrelation(corrMatrix) {
  if (!window.Plotly || !corrMatrix) return;

  const methods = ["SHAP", "LIME", "Permutation"];
  const z = methods.map(m1 => methods.map(m2 => corrMatrix[m1][m2]));

  const data = [{
    z: z,
    x: methods,
    y: methods,
    type: 'heatmap',
    colorscale: [
      [0.0, '#EF4444'],
      [0.5, '#F3F7FB'],
      [1.0, '#007BFF']
    ],
    zmin: -1,
    zmax: 1
  }];

  const layout = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#FFFFFF',
    font: { color: '#12283E', family: 'Inter' },
    margin: { l: 80, r: 20, t: 20, b: 80 },
    height: 300
  };

  Plotly.newPlot('chart-consistency-heatmap', data, layout, { responsive: true, displayModeBar: false });
}

/**
 * TAB 6: Audit History Loader
 */
async function loadHistory() {
  const refreshBtn = document.getElementById('btn-refresh-history');
  const tbody = document.getElementById('history-table-body');
  
  if (refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = '<span style="display:inline-block; animation: spin 0.6s linear infinite;">🔄</span> Refreshing...';
  }

  try {
    const res = await fetch(`${API_BASE}/history?limit=50&_t=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    if (tbody) {
      tbody.innerHTML = '';
      if (!data.predictions || data.predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-muted); padding: 25px;">No assessment queries recorded yet. Run a diagnostic test in Predict & Analyze!</td></tr>';
      } else {
        data.predictions.forEach(p => {
          const isPotable = p.prediction === 1;
          const dateStr = p.timestamp ? new Date(p.timestamp).toLocaleString() : 'N/A';
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="color: var(--text-secondary); font-size: 0.8rem;">${dateStr}</td>
            <td style="color: var(--primary-blue); font-weight: 600;">${p.model_used}</td>
            <td>${p.ph !== null && p.ph !== undefined ? parseFloat(p.ph).toFixed(2) : '-'}</td>
            <td>${p.hardness !== null && p.hardness !== undefined ? parseFloat(p.hardness).toFixed(1) : '-'}</td>
            <td>${p.solids !== null && p.solids !== undefined ? Number(Math.round(p.solids)).toLocaleString() : '-'}</td>
            <td style="font-family: 'JetBrains Mono'; font-weight: 700;">${(p.probability * 100).toFixed(1)}%</td>
            <td>
              <span class="badge-pill ${isPotable ? 'badge-potable' : 'badge-danger'}">
                ${isPotable ? 'POTABLE' : 'NON-POTABLE'}
              </span>
            </td>
          `;
          tbody.appendChild(tr);
        });
      }
    }

    if (refreshBtn) {
      refreshBtn.innerHTML = `✅ Updated (${data.total || (data.predictions ? data.predictions.length : 0)})`;
      setTimeout(() => {
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '🔄 Refresh Audit Log';
      }, 1200);
    }

  } catch (e) {
    console.error("Failed to load audit history:", e);
    if (tbody && (!tbody.children || tbody.children.length === 0)) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--danger-red); padding: 20px;">Failed to load audit log: ${e.message}</td></tr>`;
    }
    if (refreshBtn) {
      refreshBtn.innerHTML = '⚠️ Retry Refresh';
      refreshBtn.disabled = false;
    }
  }
}
