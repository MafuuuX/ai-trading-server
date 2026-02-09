/* =============================================
   AI Trading Server - Dashboard Logic
   ============================================= */

// ---- Helpers ----
function toggleSection(id) {
  const el = document.getElementById(id);
  const arrow = document.getElementById(id + 'Arrow');
  if (!el) return;
  const hidden = el.style.display === 'none';
  el.style.display = hidden ? '' : 'none';
  if (arrow) arrow.classList.toggle('open', hidden);
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status}`);
  return await res.json();
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

function formatTime(iso) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleString("de-DE"); } catch { return iso; }
}

function formatUptime(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const p = (n) => String(n).padStart(2, '0');
  if (d > 0) return `${d}d ${p(h)}:${p(m)}:${p(s)}`;
  return `${p(h)}:${p(m)}:${p(s)}`;
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

// ============================================================================
// UNIVERSAL MODEL
// ============================================================================

let lastUniStatus = {};
async function refreshUniversalStatus() {
  try {
    const data = await fetchJson("/api/training-status/universal");

    const badge = document.getElementById("uniStatusBadge");
    const vBadge = document.getElementById("uniVersionBadge");
    const progressContainer = document.getElementById("uniProgressContainer");

    // Version
    vBadge.textContent = data.version || "—";

    // Status badge
    badge.textContent = data.status || "idle";
    badge.className = "status-badge " + (data.status || "idle");

    // Progress bar
    if (data.status === "training") {
      progressContainer.style.display = "block";
      const pct = Math.round(data.progress || 0);
      document.getElementById("uniProgressFill").style.width = pct + "%";
      document.getElementById("uniProgressPct").textContent = pct + "%";
      document.getElementById("uniProgressMsg").textContent = data.message || "";
    } else {
      progressContainer.style.display = "none";
    }

    // Metrics
    const m = data.metrics || {};
    const acc = m.class_accuracy;
    document.getElementById("uniAccuracy").textContent = acc != null
      ? (acc * 100).toFixed(1) + "%" : "—";

    const mae = m.reg_mae;
    document.getElementById("uniMAE").textContent = mae != null
      ? mae.toFixed(4) : "—";

    document.getElementById("uniTickers").textContent = m.tickers_used || "—";
    document.getElementById("uniSamples").textContent = m.total_samples
      ? Number(m.total_samples).toLocaleString() : "—";

    document.getElementById("uniLastTrained").textContent = formatTime(m.trained_at);
    document.getElementById("uniDuration").textContent = formatDuration(m.duration_seconds);

  } catch (e) {
    console.error("Error loading universal status:", e);
  }
}

async function trainUniversal() {
  if (!confirm("Start universal model training? This may take 15-30 minutes.")) return;
  try {
    await fetchJson("/api/train-all", { method: "POST" });
    await refreshUniversalStatus();
  } catch (e) {
    alert(`Training failed: ${e}`);
  }
}

// ============================================================================
// HEALTH & METRICS
// ============================================================================

async function refreshHealth() {
  try {
    const data = await fetchJson("/api/health");
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Healthy";
    badge.className = "badge ok";
    document.getElementById("headerUptime").textContent = formatUptime(data.uptime_seconds);
    document.getElementById("activeModels").textContent = data.active_models;
    document.getElementById("trainingQueue").textContent = data.training_queue;
  } catch {
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Offline";
    badge.className = "badge err";
  }
}

let lastMetrics = {};
async function refreshMetrics() {
  try {
    const data = await fetchJson("/api/metrics");
    const cpuText = `${data.cpu_percent.toFixed(1)}%`;
    const ramText = `${data.ram_percent.toFixed(1)}% (${formatBytes(data.ram_used)})`;
    if (cpuText !== lastMetrics.cpuText) {
      document.getElementById("cpu").textContent = cpuText;
      lastMetrics.cpuText = cpuText;
    }
    if (ramText !== lastMetrics.ramText) {
      document.getElementById("ram").textContent = ramText;
      lastMetrics.ramText = ramText;
    }
  } catch {}
}

// ============================================================================
// PER-TICKER MODELS (Legacy)
// ============================================================================

let lastModelsData = {};
async function refreshModels() {
  const tbody = document.getElementById("modelsTable");
  if (!tbody) return;
  try {
    const models = await fetchJson("/api/models");
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No models</td></tr>';
      return;
    }
    models.forEach(m => {
      const last = lastModelsData[m.ticker];
      if (!last || last.trained_at !== m.trained_at || last.version !== m.version) {
        let row = tbody.querySelector(`tr[data-ticker="${m.ticker}"]`);
        if (!row) {
          row = document.createElement('tr');
          row.setAttribute('data-ticker', m.ticker);
          tbody.appendChild(row);
        }
        row.innerHTML = `
          <td><strong>${m.ticker}</strong></td>
          <td><span class="version-badge">${m.version}</span></td>
          <td>${formatTime(m.trained_at)}</td>
          <td>${formatBytes(m.file_size)}</td>
          <td>
            <button class="btn btn-sm btn-secondary" onclick="trainTicker('${m.ticker}')">Train</button>
            <button class="btn btn-sm btn-secondary" onclick="rollbackTicker('${m.ticker}')">Rollback</button>
          </td>`;
        lastModelsData[m.ticker] = m;
      }
    });
  } catch (e) {
    console.error('Error loading models:', e);
  }
}

// ============================================================================
// TRAINING STATUS
// ============================================================================

let lastTrainingData = {};
async function refreshTraining() {
  const tbody = document.getElementById("trainingTable");
  if (!tbody) return;
  try {
    const data = await fetchJson("/api/training-status");
    const rows = Object.values(data);
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No training data</td></tr>';
      return;
    }
    rows.forEach(r => {
      const last = lastTrainingData[r.ticker];
      if (!last || last.status !== r.status || last.progress !== r.progress || last.last_trained !== r.last_trained) {
        let row = tbody.querySelector(`tr[data-ticker="${r.ticker}"]`);
        if (!row) {
          row = document.createElement('tr');
          row.setAttribute('data-ticker', r.ticker);
          tbody.appendChild(row);
        }
        const pct = r.status === 'completed' ? 100 : (r.progress || 0);
        const cls = r.status === 'completed' ? 'complete' : (r.status === 'training' ? 'training' : 'idle');
        row.innerHTML = `
          <td><strong>${r.ticker}</strong></td>
          <td><span class="status-badge ${cls}">${r.status}</span></td>
          <td>
            <div class="progress"><div class="bar" style="width:${pct}%"></div></div>
            <div class="progress-text">${pct}%</div>
          </td>
          <td>${formatTime(r.last_trained)}</td>
          <td><button class="btn btn-sm btn-secondary" onclick="trainTicker('${r.ticker}')">Train</button></td>`;
        lastTrainingData[r.ticker] = r;
      }
    });
  } catch (e) {
    console.error('Error loading training status:', e);
  }
}

// ============================================================================
// QUEUE
// ============================================================================

let lastQueueData = {};
async function refreshQueue() {
  const queueEl = document.getElementById("queueList");
  const etaEl = document.getElementById("queueEta");
  if (!queueEl || !etaEl) return;
  try {
    const data = await fetchJson("/api/queue");
    let text = "";
    if (data.current) {
      text = `🔄 ${data.current}`;
      if (data.queue.length > 0) text += ` → ${data.queue.join(", ")}`;
    } else {
      text = data.queue.length ? data.queue.join(", ") : "Queue is empty";
    }
    const eta = data.eta_seconds ? `ETA: ~${Math.ceil(data.eta_seconds / 60)} min` : "ETA: —";
    if (text !== lastQueueData.text) { queueEl.textContent = text; lastQueueData.text = text; }
    if (eta !== lastQueueData.eta) { etaEl.textContent = eta; lastQueueData.eta = eta; }
  } catch {}
}

// ============================================================================
// LOGS
// ============================================================================

let lastLogUpdate = 0;
async function refreshLogs(force) {
  const logBox = document.getElementById("logBox");
  if (!logBox) return;
  const now = Date.now();
  if (!force && now - lastLogUpdate < 60000) return;
  lastLogUpdate = now;
  try {
    const data = await fetchJson("/api/logs");
    const lines = data.logs.map(l => `[${l.time}] ${l.message}`).join("\n");
    logBox.textContent = lines || "No logs";
  } catch {}
}

function downloadLogs() {
  const logBox = document.getElementById('logBox');
  if (!logBox) return alert('No logs available');
  const blob = new Blob([logBox.textContent || ''], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `server-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ============================================================================
// PERFORMANCE & HISTORY
// ============================================================================

let lastPerfUpdate = 0;
async function refreshPerformance() {
  const grid = document.getElementById("performanceGrid");
  if (!grid) return;
  const now = Date.now();
  if (now - lastPerfUpdate < 60000) return;
  lastPerfUpdate = now;
  try {
    const data = await fetchJson("/api/performance");
    const perf = data.performance || {};
    const entries = Object.entries(perf);
    if (!entries.length) { grid.innerHTML = '<div class="empty-state">No performance data</div>'; return; }
    grid.innerHTML = entries.map(([ticker, m]) => `
      <div class="perf-card">
        <div class="label">${ticker}</div>
        <div class="value">Acc: ${((m.class_accuracy || 0) * 100).toFixed(1)}%</div>
        <div class="value">MAE: ${(m.reg_mae || 0).toFixed(4)}</div>
      </div>
    `).join("");
  } catch {}
}

let lastHistoryUpdate = 0;
async function refreshHistory() {
  const list = document.getElementById("historyList");
  if (!list) return;
  const now = Date.now();
  if (now - lastHistoryUpdate < 60000) return;
  lastHistoryUpdate = now;
  try {
    const data = await fetchJson("/api/training-history");
    const items = data.history || [];
    if (!items.length) { list.textContent = "No history"; return; }
    list.innerHTML = items.map(h => `
      <div class="timeline-item">
        <div class="timeline-title">${h.ticker} · <span class="status-badge ${h.status === 'completed' ? 'complete' : 'failed'}">${h.status}</span></div>
        <div class="timeline-meta">
          ${formatTime(h.trained_at)} · ${formatDuration(h.duration_seconds)} · Acc: ${((h.class_accuracy || 0) * 100).toFixed(1)}% · MAE: ${(h.reg_mae || 0).toFixed(4)}
        </div>
      </div>
    `).join("");
  } catch {}
}

// ============================================================================
// ACTIONS
// ============================================================================

async function loadSectors() {
  const select = document.getElementById("sectorSelect");
  if (!select) return;
  try {
    const data = await fetchJson("/api/sectors");
    select.innerHTML = (data.sectors || []).map(s => `<option value="${s}">${s}</option>`).join("");
  } catch { select.innerHTML = "<option>Error</option>"; }
}

async function trainTicker(ticker) {
  try { await fetchJson(`/api/train/${ticker}`, { method: "POST" }); await refreshTraining(); }
  catch (e) { alert(`Training failed: ${e}`); }
}

async function rollbackTicker(ticker) {
  if (!confirm(`Rollback ${ticker}?`)) return;
  try { await fetchJson(`/api/models/${ticker}/rollback`, { method: "POST" }); await refreshModels(); }
  catch (e) { alert(`Rollback failed: ${e}`); }
}

async function trainBatch() {
  try { await fetchJson("/api/train-batch", { method: "POST" }); await refreshTraining(); }
  catch (e) { alert(`Batch training failed: ${e}`); }
}

async function trainAll() {
  if (!confirm("Queue training for ALL tickers?")) return;
  try { await fetchJson("/api/train-all-legacy", { method: "POST" }); await refreshTraining(); await refreshQueue(); }
  catch (e) { alert(`Train all failed: ${e}`); }
}

async function trainSector() {
  const sector = document.getElementById("sectorSelect").value;
  if (!sector) return;
  try { await fetchJson(`/api/train-sector/${encodeURIComponent(sector)}`, { method: "POST" }); await refreshTraining(); await refreshQueue(); }
  catch (e) { alert(`Train sector failed: ${e}`); }
}

// ============================================================================
// REINFORCEMENT LEARNING
// ============================================================================

async function refreshRLStatus() {
  try {
    const data = await fetchJson("/api/rl/config");
    const config = data.config || {};
    const status = data.status || {};
    document.getElementById("rlStatus").textContent = status.ready ? "✅ Ready" : "⏳ " + (status.reason || "Not ready");
    document.getElementById("rlTrades").textContent = status.closed_trades || 0;
    document.getElementById("rlMinTrades").textContent = config.min_trades_required || 20;
    document.getElementById("rlLastTraining").textContent = config.last_rl_training ? formatTime(config.last_rl_training) : "Never";
    document.getElementById("rlEnabled").checked = config.enabled !== false;
    document.getElementById("rlMinTradesInput").value = config.min_trades_required || 20;
  } catch { document.getElementById("rlStatus").textContent = "Error"; }
}

async function toggleRL() {
  const enabled = document.getElementById("rlEnabled").checked;
  try { await fetchJson("/api/rl/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled }) }); await refreshRLStatus(); }
  catch (e) { alert(`Failed: ${e}`); }
}

async function saveRLConfig() {
  const enabled = document.getElementById("rlEnabled").checked;
  const minTrades = parseInt(document.getElementById("rlMinTradesInput").value) || 20;
  try { await fetchJson("/api/rl/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled, min_trades_required: minTrades }) }); await refreshRLStatus(); alert("RL config saved!"); }
  catch (e) { alert(`Failed: ${e}`); }
}

async function triggerRL(force = false) {
  try {
    const url = force ? "/api/rl/trigger?force=true" : "/api/rl/trigger";
    const result = await fetchJson(url, { method: "POST" });
    alert(result.status === "queued" ? `RL queued with ${result.trade_count} trades` : `RL skipped: ${result.reason}`);
    await refreshRLStatus();
  } catch (e) { alert(`Failed: ${e}`); }
}

// ============================================================================
// SIMULATION
// ============================================================================

let lastSimData = {};
async function refreshSimStatus() {
  try {
    const data = await fetchJson("/api/simulation/status");
    const status = data.is_running ? "🔄 Running" : "✅ Idle";
    const progress = data.is_running ? `${data.progress?.toFixed(0) || 0}%` : "—";
    if (status !== lastSimData.status) { document.getElementById("simStatus").textContent = status; lastSimData.status = status; }
    if (progress !== lastSimData.progress) { document.getElementById("simProgress").textContent = progress; lastSimData.progress = progress; }
    const last = data.last_simulation;
    if (last) {
      const ret = last.total_return_pct?.toFixed(2) || "—";
      document.getElementById("simLastReturn").textContent = `${ret}%`;
      document.getElementById("simSharpe").textContent = last.sharpe_ratio?.toFixed(2) || "—";
      document.getElementById("simResultsPanel").style.display = "block";
      document.getElementById("simTotalReturn").textContent = `$${last.total_return?.toFixed(0) || 0} (${ret}%)`;
      document.getElementById("simWinRate").textContent = `${((last.win_rate || 0) * 100).toFixed(1)}%`;
      document.getElementById("simMaxDrawdown").textContent = `${last.max_drawdown_pct?.toFixed(2) || 0}%`;
      document.getElementById("simProfitFactor").textContent = last.profit_factor?.toFixed(2) || "—";
      document.getElementById("simTotalTrades").textContent = last.total_trades || 0;
      document.getElementById("simAvgHold").textContent = `${((last.avg_hold_duration_hours || 0) / 24).toFixed(1)} days`;
    } else {
      document.getElementById("simLastReturn").textContent = "—";
      document.getElementById("simSharpe").textContent = "—";
      document.getElementById("simResultsPanel").style.display = "none";
    }
  } catch (e) { console.error("Sim status error:", e); }
}

async function runSimulation() {
  try {
    const result = await fetchJson("/api/simulation/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
    if (result.status === "started") alert(`Simulation started: ${result.profile}, ${result.tickers} tickers`);
    await refreshSimStatus();
  } catch (e) { alert(e.message.includes("409") ? "Simulation already running." : `Failed: ${e}`); }
}

async function compareProfiles() {
  try { const r = await fetchJson("/api/simulation/compare", { method: "POST" }); if (r.status === "started") alert("Comparison started."); await refreshSimStatus(); }
  catch (e) { alert(e.message.includes("409") ? "Simulation running." : `Failed: ${e}`); }
}

async function optimizeLevel() {
  try { const r = await fetchJson("/api/simulation/optimize", { method: "POST" }); if (r.status === "started") alert("Optimization started."); await refreshSimStatus(); }
  catch (e) { alert(e.message.includes("409") ? "Simulation running." : `Failed: ${e}`); }
}

// ============================================================================
// REFRESH ORCHESTRATION
// ============================================================================

async function refreshCritical() {
  await Promise.all([
    refreshHealth(),
    refreshUniversalStatus(),
    refreshQueue()
  ]);
}

async function refreshMedium() {
  await Promise.all([
    refreshMetrics(),
    refreshModels(),
    refreshTraining(),
    refreshRLStatus(),
    refreshSimStatus()
  ]);
}

async function refreshSlow() {
  await Promise.all([
    refreshLogs(),
    refreshPerformance(),
    refreshHistory()
  ]);
}

// Initial load
loadSectors();
refreshCritical();
refreshMedium();
refreshSlow();

// Polling intervals
setInterval(refreshCritical, 3000);   // 3s  - health, universal, queue
setInterval(refreshMedium, 10000);    // 10s - metrics, models, training, RL, sim
setInterval(refreshSlow, 60000);      // 60s - logs, performance, history
