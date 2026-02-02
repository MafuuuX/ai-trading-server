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

async function refreshAll() {
  // Update every 3 seconds: health, training status, queue
  await Promise.all([
    refreshHealth(),
    refreshTraining(),
    refreshQueue()
  ]);
}

async function refreshInfrequent() {
  // Update every 10 seconds: metrics, models
  await Promise.all([
    refreshMetrics(),
    refreshModels()
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
