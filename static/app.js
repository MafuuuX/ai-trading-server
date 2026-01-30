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

async function refreshHealth() {
  try {
    const data = await fetchJson("/api/health");
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Healthy";
    badge.classList.remove("err");
    badge.classList.add("ok");
    document.getElementById("uptime").textContent = `${Math.floor(data.uptime_seconds)}s`;
    document.getElementById("activeModels").textContent = data.active_models;
    document.getElementById("trainingQueue").textContent = data.training_queue;
  } catch (e) {
    const badge = document.getElementById("healthBadge");
    badge.textContent = "Offline";
    badge.classList.remove("ok");
    badge.classList.add("err");
  }
}

async function refreshMetrics() {
  try {
    const data = await fetchJson("/api/metrics");
    document.getElementById("cpu").textContent = `${data.cpu_percent.toFixed(1)}%`;
    const ramText = `${data.ram_percent.toFixed(1)}% (${formatBytes(data.ram_used)} / ${formatBytes(data.ram_total)})`;
    document.getElementById("ram").textContent = ramText;
  } catch (e) {
    document.getElementById("cpu").textContent = "—";
    document.getElementById("ram").textContent = "—";
  }
}

async function refreshModels() {
  const tbody = document.getElementById("modelsTable");
  tbody.innerHTML = "<tr><td colspan='5'>Loading...</td></tr>";
  try {
    const models = await fetchJson("/api/models");
    if (!models.length) {
      tbody.innerHTML = "<tr><td colspan='5'>No models</td></tr>";
      return;
    }
    tbody.innerHTML = models.map(m => `
      <tr>
        <td>${m.ticker}</td>
        <td>${m.version}</td>
        <td>${formatTime(m.trained_at)}</td>
        <td>${formatBytes(m.file_size)}</td>
        <td>
          <button class="secondary" onclick="trainTicker('${m.ticker}')">Train</button>
          <button class="secondary" onclick="rollbackTicker('${m.ticker}')">Rollback</button>
        </td>
      </tr>
    `).join("");
  } catch (e) {
    tbody.innerHTML = "<tr><td colspan='5'>Error loading models</td></tr>";
  }
}

async function refreshTraining() {
  const tbody = document.getElementById("trainingTable");
  tbody.innerHTML = "<tr><td colspan='5'>Loading...</td></tr>";
  try {
    const data = await fetchJson("/api/training-status");
    const rows = Object.values(data);
    if (!rows.length) {
      tbody.innerHTML = "<tr><td colspan='5'>No training status</td></tr>";
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>${r.ticker}</td>
        <td>${r.status}</td>
        <td>
          <div class="progress">
            <div class="bar" style="width:${r.progress || 0}%"></div>
          </div>
          <div class="progress-text">${r.progress || 0}%</div>
        </td>
        <td>${formatTime(r.last_trained)}</td>
        <td>${formatTime(r.next_training)}</td>
        <td><button class="secondary" onclick="trainTicker('${r.ticker}')">Train</button></td>
      </tr>
    `).join("");
  } catch (e) {
    tbody.innerHTML = "<tr><td colspan='5'>Error loading status</td></tr>";
  }
}

async function refreshQueue() {
  const queueEl = document.getElementById("queueList");
  const etaEl = document.getElementById("queueEta");
  try {
    const data = await fetchJson("/api/queue");
    if (!data.queue.length) {
      queueEl.textContent = "Queue is empty";
      etaEl.textContent = "ETA: —";
      return;
    }
    queueEl.textContent = data.queue.join(", ");
    if (data.eta_seconds) {
      const mins = Math.ceil(data.eta_seconds / 60);
      etaEl.textContent = `ETA: ~${mins} min`;
    } else {
      etaEl.textContent = "ETA: —";
    }
  } catch (e) {
    queueEl.textContent = "Error loading queue";
    etaEl.textContent = "ETA: —";
  }
}

async function refreshLogs() {
  const logBox = document.getElementById("logBox");
  try {
    const data = await fetchJson("/api/logs");
    const lines = data.logs.map(l => `[${l.time}] ${l.message}`).join("\n");
    logBox.textContent = lines || "No logs";
  } catch (e) {
    logBox.textContent = "Error loading logs";
  }
}

async function refreshPerformance() {
  const grid = document.getElementById("performanceGrid");
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
    grid.innerHTML = "Error loading performance";
  }
}

async function refreshHistory() {
  const list = document.getElementById("historyList");
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
    list.textContent = "Error loading history";
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
  await refreshHealth();
  await refreshMetrics();
  await refreshModels();
  await refreshTraining();
  await refreshQueue();
  await refreshLogs();
  await refreshPerformance();
  await refreshHistory();
}

loadSectors();
refreshAll();
setInterval(refreshAll, 15000);
