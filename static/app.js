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
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

function formatUptime(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(days)}:${pad(hours)}:${pad(mins)}:${pad(secs)}`;
}

async function refreshHealth() {
  try {
    const data = await fetchJson("/api/health");
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Healthy";
    badge.classList.remove("err");
    badge.classList.add("ok");
    document.getElementById("uptime").textContent = formatUptime(data.uptime_seconds);
    document.getElementById("activeModels").textContent = data.active_models;
    document.getElementById("trainingQueue").textContent = data.training_queue;
  } catch (e) {
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Offline";
    badge.classList.remove("ok");
    badge.classList.add("err");
  }
}

let lastMetrics = {};
async function refreshMetrics() {
  try {
    const data = await fetchJson("/api/metrics");
    const cpuText = `${data.cpu_percent.toFixed(1)}%`;
    const ramText = `${data.ram_percent.toFixed(1)}% (${formatBytes(data.ram_used)} / ${formatBytes(data.ram_total)})`;
    
    if (cpuText !== lastMetrics.cpuText) {
      document.getElementById("cpu").textContent = cpuText;
      lastMetrics.cpuText = cpuText;
    }
    if (ramText !== lastMetrics.ramText) {
      document.getElementById("ram").textContent = ramText;
      lastMetrics.ramText = ramText;
    }
  } catch (e) {
    // Silent fail for metrics
  }
}

let lastModelsData = {};
async function refreshModels() {
  const tbody = document.getElementById("modelsTable");
  if (!tbody) return;
  
  try {
    const models = await fetchJson("/api/models");
    if (!models.length) {
      tbody.innerHTML = "<tr><td colspan='5'>No models</td></tr>";
      return;
    }
    
    // Only update changed rows
    models.forEach(m => {
      const lastM = lastModelsData[m.ticker];
      if (!lastM || lastM.trained_at !== m.trained_at || lastM.version !== m.version) {
        let row = tbody.querySelector(`tr[data-ticker="${m.ticker}"]`);
        if (!row) {
          row = document.createElement('tr');
          row.setAttribute('data-ticker', m.ticker);
          tbody.appendChild(row);
        }
        row.innerHTML = `
          <td>${m.ticker}</td>
          <td>${m.version}</td>
          <td>${formatTime(m.trained_at)}</td>
          <td>${formatBytes(m.file_size)}</td>
          <td>
            <button class="secondary" onclick="trainTicker('${m.ticker}')">Train</button>
            <button class="secondary" onclick="rollbackTicker('${m.ticker}')">Rollback</button>
          </td>
        `;
        lastModelsData[m.ticker] = m;
      }
    });
  } catch (e) {
    console.error('Error loading models:', e);
  }
}

let lastTrainingData = {};
async function refreshTraining() {
  const tbody = document.getElementById("trainingTable");
  if (!tbody) return;
  
  try {
    const data = await fetchJson("/api/training-status");
    const rows = Object.values(data);
    if (!rows.length) {
      tbody.innerHTML = "<tr><td colspan='6'>No training status</td></tr>";
      return;
    }
    
    // Only update rows that changed
    rows.forEach(r => {
      const lastR = lastTrainingData[r.ticker];
      if (!lastR || 
          lastR.status !== r.status || 
          lastR.progress !== r.progress ||
          lastR.last_trained !== r.last_trained) {
        
        let row = tbody.querySelector(`tr[data-ticker="${r.ticker}"]`);
        if (!row) {
          // Create new row
          row = document.createElement('tr');
          row.setAttribute('data-ticker', r.ticker);
          tbody.appendChild(row);
        }
        
        // Calculate progress: if status is 'completed', show 100%
        const displayProgress = r.status === 'completed' ? 100 : (r.progress || 0);
        const statusClass = r.status === 'completed' ? 'complete' : (r.status === 'training' ? 'training' : 'idle');
        
        row.innerHTML = `
          <td>${r.ticker}</td>
          <td><span class="status-badge ${statusClass}">${r.status}</span></td>
          <td>
            <div class="progress">
              <div class="bar" style="width:${displayProgress}%"></div>
            </div>
            <div class="progress-text">${displayProgress}%</div>
          </td>
          <td>${formatTime(r.last_trained)}</td>
          <td>${formatTime(r.next_training)}</td>
          <td><button class="secondary" onclick="trainTicker('${r.ticker}')">Train</button></td>
        `;
        lastTrainingData[r.ticker] = r;
      }
    });
  } catch (e) {
    console.error('Error loading training status:', e);
  }
}

let lastQueueData = {};
async function refreshQueue() {
  const queueEl = document.getElementById("queueList");
  const etaEl = document.getElementById("queueEta");
  const countEl = document.getElementById("queueCount");
  if (!queueEl || !etaEl) return;
  
  try {
    const data = await fetchJson("/api/queue");
    
    // Show current training + pending queue
    let queueText = "";
    if (data.current) {
      queueText = `[TRAINING] ${data.current}`;
      if (data.queue.length > 0) {
        queueText += ` → ${data.queue.join(", ")}`;
      }
    } else {
      queueText = data.queue.length ? data.queue.join(", ") : "Queue is empty";
    }
    
    const etaText = data.eta_seconds ? `ETA: ~${Math.ceil(data.eta_seconds / 60)} min` : "ETA: —";
    const countText = `${data.count} total`;
    
    if (queueText !== lastQueueData.text) {
      queueEl.textContent = queueText;
      lastQueueData.text = queueText;
    }
    if (etaText !== lastQueueData.eta) {
      etaEl.textContent = etaText;
      lastQueueData.eta = etaText;
    }
    if (countEl && countText !== lastQueueData.count) {
      countEl.textContent = countText;
      lastQueueData.count = countText;
    }
  } catch (e) {
    // Silent fail for queue
  }
}

let lastLogUpdate = 0;
async function refreshLogs() {
  const logBox = document.getElementById("logBox");
  if (!logBox) return;
  
  // Only refresh logs every 60 seconds (less frequent)
  const now = Date.now();
  if (now - lastLogUpdate < 60000) return;
  lastLogUpdate = now;
  
  try {
    const data = await fetchJson("/api/logs");
    const lines = data.logs.map(l => `[${l.time}] ${l.message}`).join("\n");
    logBox.textContent = lines || "No logs";
  } catch (e) {
    // Silent fail for logs
  }
}

// Download currently displayed logs as a .log file (user-requested feature)
function downloadLogs() {
  const logBox = document.getElementById('logBox');
  if (!logBox) return alert('No logs available');
  const text = logBox.textContent || '';
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const now = new Date();
  const stamp = now.toISOString().replace(/[:.]/g, '-');
  a.href = url;
  a.download = `server-logs-${stamp}.log`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

let lastPerfUpdate = 0;
async function refreshPerformance() {
  const grid = document.getElementById("performanceGrid");
  if (!grid) return;
  
  // Only refresh performance every 60 seconds (much less frequent)
  const now = Date.now();
  if (now - lastPerfUpdate < 60000) return;
  lastPerfUpdate = now;
  
  try {
    const data = await fetchJson("/api/performance");
    const perf = data.performance || {};
    const entries = Object.entries(perf);
    if (!entries.length) {
      grid.innerHTML = "No performance data";
      return;
    }
    grid.innerHTML = entries.map(([ticker, m]) => `
      <div class="perf-card">
        <div class="label">${ticker}</div>
        <div class="value">Acc: ${(m.class_accuracy * 100 || 0).toFixed(1)}%</div>
        <div class="value">MAE: ${(m.reg_mae || 0).toFixed(4)}</div>
      </div>
    `).join("");
  } catch (e) {
    // Silent fail for performance
  }
}

let lastHistoryUpdate = 0;
async function refreshHistory() {
  const list = document.getElementById("historyList");
  if (!list) return;
  
  // Only refresh history every 60 seconds (much less frequent)
  const now = Date.now();
  if (now - lastHistoryUpdate < 60000) return;
  lastHistoryUpdate = now;
  
  try {
    const data = await fetchJson("/api/training-history");
    const items = data.history || [];
    if (!items.length) {
      list.textContent = "No history";
      return;
    }
    list.innerHTML = items.map(h => `
      <div class="timeline-item">
        <div class="timeline-title">${h.ticker} · ${h.status}</div>
        <div class="timeline-meta">
          ${formatTime(h.trained_at)} · ${(h.duration_seconds || 0).toFixed(0)}s · Acc: ${(h.class_accuracy * 100 || 0).toFixed(1)}% · MAE: ${(h.reg_mae || 0).toFixed(4)}
        </div>
      </div>
    `).join("");
  } catch (e) {
    // Silent fail for history
  }
}

async function loadSectors() {
  const select = document.getElementById("sectorSelect");
  try {
    const data = await fetchJson("/api/sectors");
    const sectors = data.sectors || [];
    select.innerHTML = sectors.map(s => `<option value="${s}">${s}</option>`).join("");
  } catch (e) {
    select.innerHTML = "<option>Error</option>";
  }
}

async function trainTicker(ticker) {
  try {
    await fetchJson(`/api/train/${ticker}`, { method: "POST" });
    await refreshTraining();
  } catch (e) {
    alert(`Training failed: ${e}`);
  }
}

async function rollbackTicker(ticker) {
  if (!confirm(`Rollback ${ticker} to latest backup?`)) return;
  try {
    await fetchJson(`/api/models/${ticker}/rollback`, { method: "POST" });
    await refreshModels();
  } catch (e) {
    alert(`Rollback failed: ${e}`);
  }
}

async function trainBatch() {
  try {
    await fetchJson(`/api/train-batch`, { method: "POST" });
    await refreshTraining();
  } catch (e) {
    alert(`Batch training failed: ${e}`);
  }
}

async function trainAll() {
  if (!confirm("Queue training for ALL tickers?")) return;
  try {
    await fetchJson(`/api/train-all`, { method: "POST" });
    await refreshTraining();
    await refreshQueue();
  } catch (e) {
    alert(`Train all failed: ${e}`);
  }
}

async function trainSector() {
  const sector = document.getElementById("sectorSelect").value;
  if (!sector) return;
  try {
    await fetchJson(`/api/train-sector/${encodeURIComponent(sector)}`, { method: "POST" });
    await refreshTraining();
    await refreshQueue();
  } catch (e) {
    alert(`Train sector failed: ${e}`);
  }
}

// ============================================================================
// REINFORCEMENT LEARNING CONTROLS
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
  } catch (e) {
    document.getElementById("rlStatus").textContent = "Error loading";
  }
}

async function toggleRL() {
  const enabled = document.getElementById("rlEnabled").checked;
  try {
    await fetchJson("/api/rl/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled })
    });
    await refreshRLStatus();
  } catch (e) {
    alert(`Failed to toggle RL: ${e}`);
  }
}

async function saveRLConfig() {
  const enabled = document.getElementById("rlEnabled").checked;
  const minTrades = parseInt(document.getElementById("rlMinTradesInput").value) || 20;
  try {
    await fetchJson("/api/rl/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled, min_trades_required: minTrades })
    });
    await refreshRLStatus();
    alert("RL config saved!");
  } catch (e) {
    alert(`Failed to save RL config: ${e}`);
  }
}

async function triggerRL(force = false) {
  try {
    const url = force ? "/api/rl/trigger?force=true" : "/api/rl/trigger";
    const result = await fetchJson(url, { method: "POST" });
    if (result.status === "queued") {
      alert(`RL training queued with ${result.trade_count} trades`);
    } else {
      alert(`RL training skipped: ${result.reason}`);
    }
    await refreshRLStatus();
  } catch (e) {
    alert(`Failed to trigger RL training: ${e}`);
  }
}

// ============================================================================
// RISK MANAGEMENT CONTROLS
// ============================================================================

let lastRiskData = {};
async function refreshRiskProfile() {
  try {
    const data = await fetchJson("/api/risk/current");
    const profile = data.profile || {};
    const effective = data.effective_settings || {};
    
    const profileName = profile.name || "—";
    const level = profile.level || 5;
    const positionSize = effective.position_size?.toFixed(1) || profile.position_size_default?.toFixed(1) || "—";
    const stopLoss = effective.stop_loss?.toFixed(1) || profile.stop_loss_default?.toFixed(1) || "—";
    const takeProfit = effective.take_profit?.toFixed(1) || profile.take_profit_default?.toFixed(1) || "—";
    const maxTrades = effective.max_concurrent_trades || profile.max_concurrent_trades || "—";
    
    // Only update if changed
    if (profileName !== lastRiskData.profileName) {
      document.getElementById("riskProfileName").textContent = profileName;
      lastRiskData.profileName = profileName;
      updateProfileButtonHighlight(profile.name?.toLowerCase());
    }
    if (level !== lastRiskData.level) {
      document.getElementById("riskLevel").textContent = `${level}/10`;
      document.getElementById("riskLevelSlider").value = level;
      document.getElementById("riskLevelValue").textContent = level;
      lastRiskData.level = level;
      updateRiskWarning(level);
    }
    if (positionSize !== lastRiskData.positionSize) {
      document.getElementById("riskPositionSize").textContent = `${positionSize}%`;
      lastRiskData.positionSize = positionSize;
    }
    if (stopLoss !== lastRiskData.stopLoss) {
      document.getElementById("riskStopLoss").textContent = `${stopLoss}%`;
      lastRiskData.stopLoss = stopLoss;
    }
    if (takeProfit !== lastRiskData.takeProfit) {
      document.getElementById("riskTakeProfit").textContent = `${takeProfit}%`;
      lastRiskData.takeProfit = takeProfit;
    }
    if (maxTrades !== lastRiskData.maxTrades) {
      document.getElementById("riskMaxTrades").textContent = maxTrades;
      lastRiskData.maxTrades = maxTrades;
    }
  } catch (e) {
    console.error("Error loading risk profile:", e);
  }
}

function updateProfileButtonHighlight(activeProfile) {
  document.querySelectorAll('.profile-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  if (activeProfile) {
    const btn = document.querySelector(`.profile-btn.${activeProfile}`);
    if (btn) btn.classList.add('active');
  }
}

function updateRiskWarning(level) {
  const warning = document.getElementById("riskWarning");
  if (warning) {
    warning.style.display = level >= 7 ? "block" : "none";
  }
}

function updateRiskLevelDisplay(value) {
  document.getElementById("riskLevelValue").textContent = value;
  updateRiskWarning(parseInt(value));
}

async function setRiskProfile(profileName) {
  try {
    await fetchJson("/api/risk/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_name: profileName })
    });
    await refreshRiskProfile();
  } catch (e) {
    alert(`Failed to set profile: ${e}`);
  }
}

async function applyRiskLevel() {
  const level = parseInt(document.getElementById("riskLevelSlider").value);
  try {
    await fetchJson("/api/risk/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ risk_level: level })
    });
    await refreshRiskProfile();
  } catch (e) {
    alert(`Failed to apply risk level: ${e}`);
  }
}

// ============================================================================
// SIMULATION CONTROLS
// ============================================================================

let lastSimData = {};
async function refreshSimStatus() {
  try {
    const data = await fetchJson("/api/simulation/status");
    
    const status = data.is_running ? "🔄 Running" : "✅ Idle";
    const progress = data.is_running ? `${data.progress?.toFixed(0) || 0}%` : "—";
    
    if (status !== lastSimData.status) {
      document.getElementById("simStatus").textContent = status;
      lastSimData.status = status;
    }
    if (progress !== lastSimData.progress) {
      document.getElementById("simProgress").textContent = progress;
      lastSimData.progress = progress;
    }
    
    // Show last simulation results
    const last = data.last_simulation;
    if (last) {
      const returnPct = last.total_return_pct?.toFixed(2) || "—";
      const sharpe = last.sharpe_ratio?.toFixed(2) || "—";
      
      document.getElementById("simLastReturn").textContent = `${returnPct}%`;
      document.getElementById("simSharpe").textContent = sharpe;
      
      // Show results panel
      document.getElementById("simResultsPanel").style.display = "block";
      document.getElementById("simTotalReturn").textContent = `$${last.total_return?.toFixed(0) || 0} (${returnPct}%)`;
      document.getElementById("simWinRate").textContent = `${(last.win_rate * 100)?.toFixed(1) || 0}%`;
      document.getElementById("simMaxDrawdown").textContent = `${last.max_drawdown_pct?.toFixed(2) || 0}%`;
      document.getElementById("simProfitFactor").textContent = last.profit_factor?.toFixed(2) || "—";
      document.getElementById("simTotalTrades").textContent = last.total_trades || 0;
      document.getElementById("simAvgHold").textContent = `${(last.avg_hold_duration_hours / 24)?.toFixed(1) || 0} days`;
    } else {
      document.getElementById("simLastReturn").textContent = "—";
      document.getElementById("simSharpe").textContent = "—";
      document.getElementById("simResultsPanel").style.display = "none";
    }
  } catch (e) {
    console.error("Error loading simulation status:", e);
  }
}

async function runSimulation() {
  try {
    const result = await fetchJson("/api/simulation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    if (result.status === "started") {
      alert(`Simulation started: ${result.profile} profile, ${result.tickers} tickers`);
    }
    await refreshSimStatus();
  } catch (e) {
    if (e.message.includes("409")) {
      alert("A simulation is already running. Please wait.");
    } else {
      alert(`Failed to start simulation: ${e}`);
    }
  }
}

async function compareProfiles() {
  try {
    const result = await fetchJson("/api/simulation/compare", { method: "POST" });
    if (result.status === "started") {
      alert("Profile comparison started. This may take several minutes.");
    }
    await refreshSimStatus();
  } catch (e) {
    if (e.message.includes("409")) {
      alert("A simulation is already running. Please wait.");
    } else {
      alert(`Failed to compare profiles: ${e}`);
    }
  }
}

async function optimizeLevel() {
  try {
    const result = await fetchJson("/api/simulation/optimize", { method: "POST" });
    if (result.status === "started") {
      alert("Optimization started. This may take several minutes.");
    }
    await refreshSimStatus();
  } catch (e) {
    if (e.message.includes("409")) {
      alert("A simulation is already running. Please wait.");
    } else {
      alert(`Failed to optimize level: ${e}`);
    }
  }
}

async function refreshAll() {
  // Update every 3 seconds: health, training status, queue
  await Promise.all([
    refreshHealth(),
    refreshTraining(),
    refreshQueue()
  ]);
}

async function refreshInfrequent() {
  // Update every 10 seconds: metrics, models, RL status, risk, simulation
  await Promise.all([
    refreshMetrics(),
    refreshModels(),
    refreshRLStatus(),
    refreshRiskProfile(),
    refreshSimStatus()
  ]);
}

async function refreshRareLogs() {
  // Update every 30-60 seconds: logs, performance, history
  await Promise.all([
    refreshLogs(),
    refreshPerformance(),
    refreshHistory()
  ]);
}

loadSectors();
refreshAll();
refreshInfrequent();
refreshRareLogs();

setInterval(refreshAll, 3000);        // Every 3 seconds (critical updates)
setInterval(refreshInfrequent, 10000); // Every 10 seconds
setInterval(refreshRareLogs, 60000);   // Every 60 seconds
