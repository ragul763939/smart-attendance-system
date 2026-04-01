/**
 * dashboard.js
 * ─────────────
 * Client-side interactivity for the AttendAI dashboard.
 * - Live clock
 * - Auto-refresh attendance data via API
 * - Table search / filter
 */

// ── Live Clock ──────────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById("live-clock");
  if (el) {
    const now = new Date();
    el.textContent = now.toLocaleTimeString("en-IN", { hour12: false });
  }
}
setInterval(updateClock, 1000);
updateClock();


// ── Refresh Dashboard Data (API polling) ────────────────────────────────
async function refreshData() {
  try {
    // Fetch updated summary
    const summaryRes = await fetch("/api/summary");
    const summary    = await summaryRes.json();

    // Update stat cards
    setEl("stat-total",     summary.total_students);
    setEl("stat-present",   summary.present);
    setEl("stat-absent",    summary.absent);
    setEl("stat-attentive", summary.attentive);
    setEl("stat-distracted",summary.distracted);

    // Update progress bars
    const pct  = summary.total_students > 0
      ? ((summary.present / summary.total_students) * 100).toFixed(1) : 0;
    const apct = summary.present > 0
      ? ((summary.attentive / summary.present) * 100).toFixed(1) : 0;

    setStyle("progress-fill",  `width: ${pct}%`);
    setStyle("attention-fill", `width: ${apct}%`);
    setEl("bar-pct",  `${pct}%`);
    setEl("attn-pct", summary.present > 0 ? `${apct}%` : "—");

    // Fetch and re-render attendance table
    const attRes    = await fetch("/api/attendance");
    const records   = await attRes.json();
    renderTable(records);

  } catch (err) {
    console.warn("[Dashboard] Refresh error:", err);
  }
}

// ── Render attendance table rows from JSON ───────────────────────────────
function renderTable(records) {
  const tbody = document.getElementById("attendance-body");
  if (!tbody) return;

  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-row">No attendance records for today yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map((row, i) => `
    <tr>
      <td class="mono">${i + 1}</td>
      <td class="mono">${row.student_id}</td>
      <td><strong>${row.name}</strong></td>
      <td class="mono">${row.time}</td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:${row.attention_score}%"></div>
          <span class="score-num">${row.attention_score}%</span>
        </div>
      </td>
      <td>
        <span class="badge ${row.status === 'Attentive' ? 'badge-green' : 'badge-orange'}">
          ${row.status}
        </span>
      </td>
    </tr>
  `).join("");
}


// ── Table Search / Filter ────────────────────────────────────────────────
function filterTable(query) {
  const table = document.getElementById("attendance-table");
  if (!table) return;

  const rows = table.querySelectorAll("tbody tr");
  const q    = query.toLowerCase().trim();

  rows.forEach(row => {
    const text = row.textContent.toLowerCase();
    row.style.display = (!q || text.includes(q)) ? "" : "none";
  });
}


// ── Utility Helpers ──────────────────────────────────────────────────────
function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setStyle(id, style) {
  const el = document.getElementById(id);
  if (el) el.setAttribute("style", style);
}


// ── Auto-refresh every 15 seconds ───────────────────────────────────────
setInterval(refreshData, 15000);
