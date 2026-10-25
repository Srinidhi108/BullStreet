/**
 * BullStreet Bloomberg Terminal Interactive Engine (bloomberg.js)
 */

// Terminal State
let currentIpoId = "IPO_0816"; // Default starting IPO
let currentIpoData = null;
let currentPrediction = null;
let soundEnabled = true;

// Chart Instances
let shapChart = null;
let trajectoryChart = null;
let subscriptionChart = null;

// Audio Synth for Authentic Terminal Beep
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playTerminalBeep(freq = 880, type = "square", duration = 0.04) {
  if (!soundEnabled) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    // Ignore audio context errors if blocked by browser
  }
}

// Clock updates
function updateClock() {
  const now = new Date();
  const istStr = now.toLocaleTimeString("en-US", {
    timeZone: "Asia/Kolkata",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const dateStr = now.toISOString().slice(0, 10);
  document.getElementById("terminalClock").innerText = `${dateStr} ${istStr} IST`;
}
setInterval(updateClock, 1000);
updateClock();

// Initialize Terminal
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await loadMarketPulse();
  await loadBenchmarks();
  await searchIpos("");
  await selectIpo(currentIpoId);
});

function setupEventListeners() {
  // Command Bar Input & GO Button
  const cmdInput = document.getElementById("cmdInput");
  const goBtn = document.getElementById("goBtn");

  const executeCommand = () => {
    playTerminalBeep(920);
    const val = cmdInput.value.trim().toUpperCase();
    if (val.startsWith("IPO_") || val.length >= 3) {
      selectIpo(val);
    }
  };

  goBtn.addEventListener("click", executeCommand);
  cmdInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      executeCommand();
    }
  });

  // Search filter input
  const searchInput = document.getElementById("searchInput");
  searchInput.addEventListener("input", (e) => {
    searchIpos(e.target.value);
  });

  // Function Keys
  document.querySelectorAll(".bbg-fkey").forEach((btn) => {
    btn.addEventListener("click", () => {
      playTerminalBeep(750);
      document.querySelectorAll(".bbg-fkey").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      showTab(tab);
    });
  });

  // Audio Toggle
  const audioToggle = document.getElementById("audioToggle");
  if (audioToggle) {
    audioToggle.addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      audioToggle.classList.toggle("active", soundEnabled);
      audioToggle.innerText = soundEnabled ? "[AUDIO: ON]" : "[AUDIO: MUTE]";
      if (soundEnabled) playTerminalBeep(600);
    });
  }

  // Simulator Form Submit
  const simForm = document.getElementById("simulatorForm");
  if (simForm) {
    simForm.addEventListener("submit", (e) => {
      e.preventDefault();
      runCustomSimulation();
    });
  }
}

function showTab(tabName) {
  const views = ["overviewView", "peersView", "benchmarksView", "simulatorView"];
  views.forEach((v) => {
    const el = document.getElementById(v);
    if (el) el.style.display = "none";
  });

  if (tabName === "overview" || tabName === "predict" || tabName === "charts" || tabName === "shap") {
    document.getElementById("overviewView").style.display = "block";
    if (tabName === "charts") {
      document.getElementById("chartsSection").scrollIntoView({ behavior: "smooth" });
    } else if (tabName === "shap") {
      document.getElementById("shapSection").scrollIntoView({ behavior: "smooth" });
    }
  } else if (tabName === "peers") {
    document.getElementById("peersView").style.display = "block";
  } else if (tabName === "benchmarks") {
    document.getElementById("benchmarksView").style.display = "block";
  } else if (tabName === "simulator") {
    document.getElementById("simulatorView").style.display = "block";
    populateSimulatorDefaults();
  }
}

// 1. Load Market Telemetry Pulse
async function loadMarketPulse() {
  try {
    const res = await fetch("/api/market-pulse");
    const data = await res.json();
    if (data.total_ipos) {
      document.getElementById("telemetryTotal").innerText = data.total_ipos;
      document.getElementById("telemetryWinRate").innerText = `${data.overall_win_rate_pct}%`;
      document.getElementById("telemetryAvgRet").innerText = `+${data.avg_listing_return_pct}%`;
    }
  } catch (e) {
    console.error("Failed to load market pulse", e);
  }
}

// 2. Load Model Validation Benchmarks
async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    const bench = await res.json();
    if (!bench.classification) return;

    // Top ticker telemetry
    document.getElementById("telemetryAuc").innerText = bench.classification.rf_auc.toFixed(3);
    document.getElementById("telemetryR2").innerText = bench.regression.r2.toFixed(3);

    // Benchmarks Tab Populate
    const body = document.getElementById("benchmarkTableBody");
    if (body) {
      body.innerHTML = `
        <tr>
          <td style="color:var(--bbg-cyan);">Always-Positive Naive Baseline</td>
          <td>-</td>
          <td>${(bench.baseline_always_positive_acc * 100).toFixed(1)}%</td>
          <td>0.500</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="color:var(--bbg-amber); font-weight:bold;">Random Forest Classifier</td>
          <td style="color:var(--bbg-green); font-weight:bold;">${bench.classification.rf_auc.toFixed(3)}</td>
          <td>${(bench.classification.accuracy * 100).toFixed(1)}%</td>
          <td>${(bench.classification.precision * 100).toFixed(1)}%</td>
          <td>${(bench.classification.recall * 100).toFixed(1)}%</td>
          <td>${bench.classification.f1.toFixed(3)}</td>
          <td style="color:var(--bbg-green);">OUTPERFORMS</td>
        </tr>
        <tr>
          <td style="color:var(--bbg-amber); font-weight:bold;">Dual Ensemble Regression (LGBM+RF)</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td style="color:var(--bbg-cyan);">MAE: ${bench.regression.mae}%</td>
          <td style="color:var(--bbg-cyan);">R²: ${bench.regression.r2.toFixed(3)}</td>
          <td style="color:var(--bbg-green);">Dir Acc: ${(bench.regression.directional_accuracy * 100).toFixed(1)}%</td>
        </tr>
      `;
    }
  } catch (e) {
    console.error("Failed to load benchmarks", e);
  }
}

// 3. Search and List IPOs
async function searchIpos(query) {
  try {
    const res = await fetch(`/api/ipos?limit=60&q=${encodeURIComponent(query)}`);
    const ipos = await res.json();
    const listEl = document.getElementById("ipoList");
    listEl.innerHTML = "";

    ipos.forEach((ipo) => {
      const item = document.createElement("div");
      item.className = `bbg-ipo-item ${ipo.ipo_id === currentIpoId ? "active" : ""}`;
      item.onclick = () => {
        playTerminalBeep(800);
        selectIpo(ipo.ipo_id);
      };

      const isPositive = ipo.actual_return_pct >= 0;
      const retClass = isPositive ? "bbg-item-return" : "bbg-item-return negative";
      const retSign = isPositive ? "+" : "";

      item.innerHTML = `
        <div class="bbg-item-row1">
          <span class="bbg-item-ticker">${ipo.ipo_id} [${ipo.company_symbol || "N/A"}]</span>
          <span class="${retClass}">${retSign}${ipo.actual_return_pct}%</span>
        </div>
        <div class="bbg-item-row2">
          <span>${ipo.company_name.slice(0, 24)}</span>
          <span>${ipo.total_subscription}x</span>
        </div>
      `;
      listEl.appendChild(item);
    });
  } catch (e) {
    console.error("Failed to search IPOs", e);
  }
}

// 4. Select and Display an IPO
async function selectIpo(ipoId) {
  try {
    currentIpoId = ipoId;
    document.getElementById("cmdInput").value = `${ipoId} EQUITY <GO>`;

    // Update active highlight in list
    document.querySelectorAll(".bbg-ipo-item").forEach((el) => {
      if (el.innerText.includes(ipoId)) {
        el.classList.add("active");
      } else {
        el.classList.remove("active");
      }
    });

    // Fetch IPO Detail & Model Prediction
    const [detailRes, predRes, peersRes] = await Promise.all([
      fetch(`/api/ipos/${ipoId}`),
      fetch(`/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ipo_id: ipoId }),
      }),
      fetch(`/api/peers/${ipoId}?n=5`),
    ]);

    if (!detailRes.ok) {
      alert(`IPO ${ipoId} not found.`);
      return;
    }

    currentIpoData = await detailRes.json();
    currentPrediction = await predRes.json();
    const peersData = await peersRes.json();

    renderSecurityHeader(currentIpoData);
    renderPredictionResults(currentIpoData, currentPrediction);
    renderShapWaterfall(currentPrediction.shap_explanation);
    renderCharts(currentIpoData, currentPrediction);
    renderPeersTable(peersData);
  } catch (e) {
    console.error(`Error selecting IPO ${ipoId}`, e);
  }
}

// Render Header Security Metadata
function renderSecurityHeader(data) {
  document.getElementById("secTitle").innerText = `${data.company_name.toUpperCase()} [${data.company_symbol || data.ipo_id}]`;
  document.getElementById("secSegment").innerText = `${data.listing_segment.toUpperCase()} | ${data.exchange || "NSE/BSE"}`;
  document.getElementById("metaSector").innerText = data.sector || "General Industry";
  document.getElementById("metaDates").innerText = `${data.ipo_open_date} - ${data.ipo_close_date}`;
  document.getElementById("metaListDate").innerText = data.listing_date || "N/A";
  document.getElementById("metaIssuePrice").innerText = `₹${data.issue_price || 0}`;
  document.getElementById("metaIssueSize").innerText = `₹${data.total_issue_size_cr || 0} Cr`;
  document.getElementById("metaLotSize").innerText = `${data.lot_size || "N/A"} shares`;
  document.getElementById("metaTotalSub").innerText = `${data.total_subscription || 0}x`;
  document.getElementById("metaQibSub").innerText = `${data.qib_subscription || 0}x`;
  document.getElementById("metaNiiSub").innerText = `${data.nii_subscription || 0}x`;
  document.getElementById("metaRetailSub").innerText = `${data.retail_subscription || 0}x`;
  document.getElementById("metaRevenue").innerText = `₹${data.revenue_cr || "N/A"} Cr`;
  document.getElementById("metaPat").innerText = `₹${data.pat_cr || "N/A"} Cr (${data.pat_margin_pct || 0}%)`;
}

// Render Prediction Outputs
function renderPredictionResults(data, pred) {
  const cls = pred.classification;
  const reg = pred.regression;

  // Classification Readout
  const winEl = document.getElementById("predWinProb");
  winEl.innerText = `${cls.win_probability_pct}%`;
  winEl.className = cls.win_probability_pct >= 50 ? "bbg-big-metric green" : "bbg-big-metric red";

  const verdictEl = document.getElementById("predVerdict");
  verdictEl.innerText = cls.verdict;
  verdictEl.className = cls.win_probability_pct >= 50 ? "bbg-verdict-pill buy" : "bbg-verdict-pill avoid";

  document.getElementById("predRiskBadge").innerText = cls.risk_category;
  document.getElementById("predConfidence").innerText = `${cls.confidence} (${cls.model_agreement_pct}% consensus)`;

  // Regression Readout
  const retEl = document.getElementById("predExpectedReturn");
  const isRetPos = reg.expected_listing_return_pct >= 0;
  retEl.innerText = `${isRetPos ? "+" : ""}${reg.expected_listing_return_pct}%`;
  retEl.className = isRetPos ? "bbg-big-metric green" : "bbg-big-metric red";

  document.getElementById("predClosePrice").innerText = `₹${reg.projected_close_price.toFixed(2)}`;
  document.getElementById("predOpenReturn").innerText = `${reg.expected_opening_gain_pct >= 0 ? "+" : ""}${reg.expected_opening_gain_pct}%`;
  document.getElementById("predConfInterval").innerText = `[ ${reg.confidence_interval_pct[0]}% to ${reg.confidence_interval_pct[1]}% ]`;

  // Ground Truth (Actual Listing Day Result if available)
  const groundTruthCard = document.getElementById("groundTruthCard");
  if (data.listing_day_close_return_pct !== null && data.listing_day_close_return_pct !== undefined) {
    groundTruthCard.style.display = "block";
    const actRet = parseFloat(data.listing_day_close_return_pct);
    const actIsPos = actRet >= 0;
    const actEl = document.getElementById("actualReturnVal");
    actEl.innerText = `${actIsPos ? "+" : ""}${actRet.toFixed(2)}%`;
    actEl.style.color = actIsPos ? "var(--bbg-green)" : "var(--bbg-red)";

    const delta = (actRet - reg.expected_listing_return_pct).toFixed(2);
    document.getElementById("actualDeltaVal").innerText = `Delta vs Pred: ${delta >= 0 ? "+" : ""}${delta}%`;
  } else {
    groundTruthCard.style.display = "none";
  }
}

// Render SHAP Waterfall Factors
function renderShapWaterfall(shapList) {
  const container = document.getElementById("shapList");
  container.innerHTML = "";

  if (!shapList || shapList.length === 0) {
    container.innerHTML = "<div style='color:var(--bbg-gray-muted)'>No SHAP data available.</div>";
    return;
  }

  // Find max absolute impact to scale bars
  const maxImpact = Math.max(...shapList.map((f) => Math.abs(f.impact)), 1.0);

  shapList.forEach((item) => {
    const isPos = item.direction === "POSITIVE";
    const widthPct = Math.min((Math.abs(item.impact) / maxImpact) * 50, 50);

    const row = document.createElement("div");
    row.className = "bbg-shap-row";

    // Clean feature name display
    const label = item.feature.replace(/_/g, " ").toUpperCase();
    const sign = isPos ? "+" : "";
    const colorClass = isPos ? "positive" : "negative";

    row.innerHTML = `
      <div class="bbg-shap-feat" title="${item.feature}">${label}</div>
      <div class="bbg-shap-bar-container">
        <div class="bbg-shap-bar ${colorClass}" style="width: ${widthPct}%;"></div>
      </div>
      <div class="bbg-shap-val ${colorClass}">${sign}${item.impact}%</div>
    `;
    container.appendChild(row);
  });
}

// Render Financial Visualizations (Trajectory & Subscription Breakdown)
function renderCharts(data, pred) {
  const reg = pred.regression;
  const h = reg.horizon_forecast;
  const issuePrice = parseFloat(data.issue_price || 100);

  // 1. Multi-Horizon Forecast Line Chart
  const ctxTraj = document.getElementById("trajectoryChartCanvas").getContext("2d");
  if (trajectoryChart) trajectoryChart.destroy();

  const labels = ["Issue Price", "Day 1 Open", "Day 1 Close", "Session 7 Proj", "Session 30 Proj"];
  const pricePoints = [
    issuePrice,
    reg.projected_open_price,
    reg.projected_close_price,
    issuePrice * (1 + h.session_7_pct / 100),
    issuePrice * (1 + h.session_30_pct / 100),
  ];

  trajectoryChart = new Chart(ctxTraj, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Price Target (₹)",
          data: pricePoints,
          borderColor: "#ff9900",
          backgroundColor: "rgba(255, 153, 0, 0.1)",
          borderWidth: 2,
          pointBackgroundColor: "#00e5ff",
          pointBorderColor: "#ffffff",
          pointRadius: 4,
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0d1017",
          borderColor: "#ff9900",
          borderWidth: 1,
          titleColor: "#ff9900",
          bodyColor: "#ffffff",
        },
      },
      scales: {
        x: {
          ticks: { color: "#647087", font: { family: "Consolas" } },
          grid: { color: "#161b26" },
        },
        y: {
          ticks: { color: "#c0c6d4", font: { family: "Consolas" } },
          grid: { color: "#161b26" },
        },
      },
    },
  });

  // 2. Subscription Breakdown Bar Chart
  const ctxSub = document.getElementById("subscriptionChartCanvas").getContext("2d");
  if (subscriptionChart) subscriptionChart.destroy();

  const subLabels = ["QIB", "NII", "Retail", "Total"];
  const subVals = [
    parseFloat(data.qib_subscription || 0),
    parseFloat(data.nii_subscription || 0),
    parseFloat(data.retail_subscription || 0),
    parseFloat(data.total_subscription || 0),
  ];

  subscriptionChart = new Chart(ctxSub, {
    type: "bar",
    data: {
      labels: subLabels,
      datasets: [
        {
          label: "Subscription (x)",
          data: subVals,
          backgroundColor: ["#00e5ff", "#ff9900", "#00ff66", "#ffffff"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: "#c0c6d4", font: { family: "Consolas" } },
          grid: { display: false },
        },
        y: {
          ticks: { color: "#647087", font: { family: "Consolas" } },
          grid: { color: "#161b26" },
        },
      },
    },
  });
}

// Render Relative Valuation Historical Peers
function renderPeersTable(peers) {
  const tbody = document.getElementById("peersTableBody");
  tbody.innerHTML = "";

  if (!peers || peers.length === 0) {
    tbody.innerHTML = "<tr><td colspan='7'>No peer records found.</td></tr>";
    return;
  }

  peers.forEach((p) => {
    const tr = document.createElement("tr");
    const isWin = p.actual_listing_return_pct >= 0;
    const retColor = isWin ? "var(--bbg-green)" : "var(--bbg-red)";
    const sign = isWin ? "+" : "";

    tr.innerHTML = `
      <td style="color:var(--bbg-cyan); font-weight:bold; cursor:pointer;" onclick="selectIpo('${p.ipo_id}')">${p.ipo_id}</td>
      <td>${p.company_name}</td>
      <td>${p.sector}</td>
      <td>₹${p.issue_price}</td>
      <td>₹${p.total_issue_size_cr} Cr</td>
      <td>${p.total_subscription}x</td>
      <td style="color:${retColor}; font-weight:bold;">${sign}${p.actual_listing_return_pct}%</td>
    `;
    tbody.appendChild(tr);
  });
}

// Populate Simulator with current IPO values
function populateSimulatorDefaults() {
  if (!currentIpoData) return;
  document.getElementById("simPrice").value = currentIpoData.issue_price || 250;
  document.getElementById("simSize").value = currentIpoData.total_issue_size_cr || 500;
  document.getElementById("simSub").value = currentIpoData.total_subscription || 15;
  document.getElementById("simQib").value = currentIpoData.qib_subscription || 20;
  document.getElementById("simRetail").value = currentIpoData.retail_subscription || 10;
  document.getElementById("simRevenue").value = currentIpoData.revenue_cr || 300;
  document.getElementById("simPat").value = currentIpoData.pat_cr || 45;
  document.getElementById("simVix").value = currentIpoData.india_vix || 14.5;
}

// Run Custom What-If Simulation
async function runCustomSimulation() {
  playTerminalBeep(990);
  const payload = {
    features: {
      company_name: "Simulated Custom Issue",
      listing_segment: "Mainboard",
      exchange: "NSE, BSE",
      issue_price: parseFloat(document.getElementById("simPrice").value),
      total_issue_size_cr: parseFloat(document.getElementById("simSize").value),
      total_subscription: parseFloat(document.getElementById("simSub").value),
      qib_subscription: parseFloat(document.getElementById("simQib").value),
      retail_subscription: parseFloat(document.getElementById("simRetail").value),
      revenue_cr: parseFloat(document.getElementById("simRevenue").value),
      pat_cr: parseFloat(document.getElementById("simPat").value),
      india_vix: parseFloat(document.getElementById("simVix").value),
      nifty_return_5d: 0.8,
      nifty_return_20d: 2.1,
      nifty_return_60d: 5.4,
      lot_size: 50,
      sector: "Technology",
    },
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const pred = await res.json();

    // Render results in simulator view
    const simResEl = document.getElementById("simResultsBox");
    simResEl.style.display = "block";

    const isPos = pred.regression.expected_listing_return_pct >= 0;
    simResEl.innerHTML = `
      <div style="font-size:14px; font-weight:bold; color:var(--bbg-amber); margin-bottom:8px;">
        SIMULATION RESULTS [WHAT-IF ENGINE]
      </div>
      <div style="display:flex; gap:20px; flex-wrap:wrap;">
        <div>
          <div style="font-size:10px; color:var(--bbg-gray-muted);">WIN PROBABILITY</div>
          <div style="font-size:28px; font-weight:bold; color:var(--bbg-green);">${pred.classification.win_probability_pct}%</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--bbg-gray-muted);">EXPECTED LISTING RETURN</div>
          <div style="font-size:28px; font-weight:bold; color:${isPos ? "var(--bbg-green)" : "var(--bbg-red)"};">
            ${isPos ? "+" : ""}${pred.regression.expected_listing_return_pct}%
          </div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--bbg-gray-muted);">PROJECTED CLOSE PRICE</div>
          <div style="font-size:28px; font-weight:bold; color:var(--bbg-white);">₹${pred.regression.projected_close_price}</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--bbg-gray-muted);">RECOMMENDATION</div>
          <div style="font-size:20px; font-weight:bold; color:var(--bbg-cyan); margin-top:4px;">${pred.classification.verdict}</div>
        </div>
      </div>
    `;
  } catch (e) {
    console.error("Simulation failed", e);
  }
}
