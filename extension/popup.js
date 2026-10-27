/**
 * Logic điều khiển Tầng 3: Action Popup — VN Anti-Clickbait Shield
 */

document.addEventListener("DOMContentLoaded", async () => {
  const toggleEnabled = document.getElementById("toggle-enabled");
  const toggleBlur = document.getElementById("toggle-blur");
  const toggleHover = document.getElementById("toggle-hover");
  const toggleReadingPill = document.getElementById("toggle-reading-pill");
  const sliderSensitivity = document.getElementById("slider-sensitivity");
  const sensitivityLabel = document.getElementById("sensitivity-label");

  const statCbCount = document.getElementById("stat-cb-count");
  const statTimeSaved = document.getElementById("stat-time-saved");
  const engineStatusText = document.getElementById("engine-status-text");

  const cpDomain = document.getElementById("cp-domain");
  const cpCleanScore = document.getElementById("cp-clean-score");
  const cpDetails = document.getElementById("cp-details");
  const btnRescan = document.getElementById("btn-rescan");
  const btnOpenDemo = document.getElementById("btn-open-demo");
  const btnResetStats = document.getElementById("btn-reset-stats");

  // Quick Checker elements
  const quickTitle = document.getElementById("quick-title");
  const quickLead = document.getElementById("quick-lead");
  const btnQuickCheck = document.getElementById("btn-quick-check");
  const quickResultBox = document.getElementById("quick-result-box");

  // 1. Chuyển đổi Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // 2. Nạp cấu hình & thống kê từ chrome.storage.local
  const stored = await chrome.storage.local.get({
    enabled: true,
    blurClickbait: false,
    showHoverSummary: true,
    showReadingAssistant: true,
    sensitivity: 0.50,
    apiUrl: "http://127.0.0.1:7860",
    stats: { totalScanned: 0, clickbaitDetected: 0, minutesSaved: 0 }
  });

  toggleEnabled.checked = Boolean(stored.enabled);
  toggleBlur.checked = Boolean(stored.blurClickbait);
  toggleHover.checked = Boolean(stored.showHoverSummary);
  toggleReadingPill.checked = Boolean(stored.showReadingAssistant);
  sliderSensitivity.value = Number(stored.sensitivity || 0.50);
  updateSensitivityLabel(Number(stored.sensitivity || 0.50));
  renderStats(stored.stats);

  // 3. Kiểm tra trạng thái Backend AI Server (Ensemble vs Local ML)
  chrome.runtime.sendMessage({ type: "CHECK_BACKEND" }, (resp) => {
    if (resp && resp.online) {
      engineStatusText.innerHTML = `<span style="color:#4ade80;font-weight:700;">● AI Server Online (Ensemble 4 Models)</span>`;
    } else {
      engineStatusText.innerHTML = `<span style="color:#38bdf8;font-weight:700;">● Local ML Engine (Siêu tốc · Offline)</span>`;
    }
  });

  // 4. Lấy thông tin thống kê của Tab báo điện tử đang mở
  queryActiveTabSummary();

  function updateSensitivityLabel(val) {
    const pct = Math.round(val * 100);
    let modeText = "Cân bằng";
    if (val <= 0.42) modeText = "Nghiêm ngặt";
    else if (val >= 0.60) modeText = "Thoải mái";
    sensitivityLabel.textContent = `${modeText} (${pct}%)`;
  }

  function renderStats(stats) {
    const cb = (stats && stats.clickbaitDetected) || 0;
    const mins = (stats && stats.minutesSaved) || 0;
    statCbCount.textContent = `${cb} bài`;
    statTimeSaved.textContent = `~${mins} phút`;
  }

  async function saveAndBroadcastSettings() {
    const newSettings = {
      enabled: toggleEnabled.checked,
      blurClickbait: toggleBlur.checked,
      showHoverSummary: toggleHover.checked,
      showReadingAssistant: toggleReadingPill.checked,
      sensitivity: Number(sliderSensitivity.value)
    };
    await chrome.storage.local.set(newSettings);

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id) {
      chrome.tabs.sendMessage(
        tab.id,
        { type: "SETTINGS_UPDATED", settings: newSettings },
        () => {
          void chrome.runtime.lastError;
          setTimeout(queryActiveTabSummary, 350);
        }
      );
    }
  }

  toggleEnabled.addEventListener("change", saveAndBroadcastSettings);
  toggleBlur.addEventListener("change", saveAndBroadcastSettings);
  toggleHover.addEventListener("change", saveAndBroadcastSettings);
  toggleReadingPill.addEventListener("change", saveAndBroadcastSettings);
  sliderSensitivity.addEventListener("input", () => {
    updateSensitivityLabel(Number(sliderSensitivity.value));
  });
  sliderSensitivity.addEventListener("change", saveAndBroadcastSettings);

  async function queryActiveTabSummary() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) return;

    chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_SUMMARY" }, (resp) => {
      if (chrome.runtime.lastError || !resp || !resp.ok) {
        cpDomain.textContent = "🌐 Trang ngoài danh sách tự động";
        cpCleanScore.textContent = "Chưa quét";
        cpDetails.textContent =
          "Hãy mở một báo điện tử Việt Nam (VnExpress, Dân Trí, Tuổi Trẻ, Kênh14...) hoặc nhấn 'Trang báo thử nghiệm' bên dưới.";
        return;
      }

      const ps = resp.pageStats;
      cpDomain.textContent = `🌐 ${ps.domain}`;
      if (ps.totalScanned > 0) {
        const cleanRatio = Math.round(
          ((ps.totalScanned - ps.clickbaitCount) / ps.totalScanned) * 100
        );
        cpCleanScore.textContent = `Độ sạch: ${cleanRatio}%`;
        cpDetails.innerHTML = `Đã quét <b>${ps.totalScanned}</b> tiêu đề · Phát hiện <b style="color:#f87171;">${ps.clickbaitCount} tin giật tít</b> và <b style="color:#fbbf24;">${ps.yellowCount} tin gây tò mò</b>.`;
      } else {
        cpCleanScore.textContent = "0 bài";
        cpDetails.textContent = "Không tìm thấy danh sách tiêu đề bài báo nào trên trang này.";
      }
    });

    // Cập nhật lại tổng thống kê
    const latest = await chrome.storage.local.get("stats");
    if (latest && latest.stats) renderStats(latest.stats);
  }

  btnRescan.addEventListener("click", saveAndBroadcastSettings);

  btnOpenDemo.addEventListener("click", () => {
    chrome.tabs.create({ url: "http://127.0.0.1:7860/demo-news" });
  });

  btnResetStats.addEventListener("click", async () => {
    const zeroStats = { totalScanned: 0, clickbaitDetected: 0, minutesSaved: 0 };
    await chrome.storage.local.set({ stats: zeroStats });
    renderStats(zeroStats);
  });

  // 5. Tính năng Kiểm tra nhanh văn bản (Tab 2)
  btnQuickCheck.addEventListener("click", () => {
    const title = (quickTitle.value || "").trim();
    const lead = (quickLead.value || "").trim();
    if (!title) {
      quickResultBox.classList.remove("hidden");
      quickResultBox.innerHTML = `<span style="color:#f87171;">Vui lòng nhập tiêu đề bài báo cần kiểm tra.</span>`;
      return;
    }

    btnQuickCheck.textContent = "⏳ Đang phân tích...";
    chrome.runtime.sendMessage(
      {
        type: "PREDICT_SINGLE",
        title,
        lead_paragraph: lead
      },
      (resp) => {
        btnQuickCheck.textContent = "🚀 Phân tích ngay";
        if (!resp || !resp.ok || !resp.result) return;

        const r = resp.result;
        const pct = Math.round(r.prob_clickbait * 100);
        let color = "#4ade80";
        let icon = "🟢";
        if (r.badge_level === "red") {
          color = "#f87171";
          icon = "🔴";
        } else if (r.badge_level === "yellow") {
          color = "#fbbf24";
          icon = "🟡";
        }

        const trig = (r.trigger_words || []).map(t => `"${t.word}"`).join(", ") || "Không có";
        const reason = (r.reasons_vi && r.reasons_vi[0]) || "";

        quickResultBox.classList.remove("hidden");
        quickResultBox.innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b style="color:${color};font-size:13px;">${icon} ${r.verdict_vi}</b>
            <b style="color:${color};">${pct}%</b>
          </div>
          <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">• ${reason}</div>
          <div style="font-size:11.5px;color:#94a3b8;margin-bottom:6px;"><b>⚡ Từ khóa:</b> ${trig}</div>
          ${
            r.is_clickbait && r.declickbait_title
              ? `<div style="background:#0f172a;padding:7px 9px;border-radius:6px;border-left:3px solid #38bdf8;font-size:11.5px;">
                   <b style="color:#38bdf8;">✨ Gợi ý tháo tít:</b> “${r.declickbait_title}”
                 </div>`
              : ""
          }
        `;
      }
    );
  });
});
