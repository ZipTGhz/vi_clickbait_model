/**
 * Content Script — VN Anti-Clickbait Shield
 * Thực thi Tầng 1 (In-Page Ambient Badges & Hover Tooltip) và Tầng 2 (Reading Assistant Floating Pill)
 * trên các báo điện tử Việt Nam (VnExpress, Dân Trí, Tuổi Trẻ, Thanh Niên, Báo Mới, 24h, Kênh14, SaoStar...)
 * và trang báo thử nghiệm (/demo-news).
 */

(() => {
  if (window.__VN_ACB_INITIALIZED__) return;
  window.__VN_ACB_INITIALIZED__ = true;

  let currentSettings = {
    enabled: true,
    blurClickbait: false,
    showHoverSummary: true,
    showReadingAssistant: true,
    sensitivity: 0.50,
    whitelistedDomains: []
  };

  // Lưu trữ kết quả phân tích của các tiêu đề trên trang hiện tại
  const pageArticlesMap = new Map(); // id -> { el, titleEl, originalTitle, linkUrl, leadText, prediction, badgeEl }
  let pageStats = {
    domain: window.location.hostname.replace(/^www\./, "") || "localhost",
    totalScanned: 0,
    clickbaitCount: 0,
    yellowCount: 0,
    cleanCount: 0,
    engineMode: "local_ml"
  };

  let tooltipEl = null;
  let tooltipHideTimer = null;
  let readingPillEl = null;

  /**
   * 1. KHỞI TẠO HOVER TOOLTIP DÙNG CHUNG TRÊN TRANG
   */
  function ensureTooltipElement() {
    if (tooltipEl) return tooltipEl;
    tooltipEl = document.createElement("div");
    tooltipEl.className = "vn-acb-tooltip";
    tooltipEl.style.display = "none";

    tooltipEl.addEventListener("mouseenter", () => {
      if (tooltipHideTimer) clearTimeout(tooltipHideTimer);
    });
    tooltipEl.addEventListener("mouseleave", () => {
      hideTooltip();
    });

    document.body.appendChild(tooltipEl);
    return tooltipEl;
  }

  function showTooltipForArticle(articleId, anchorEl) {
    if (!currentSettings.showHoverSummary) return;
    const item = pageArticlesMap.get(articleId);
    if (!item || !item.prediction) return;

    if (tooltipHideTimer) clearTimeout(tooltipHideTimer);
    const tt = ensureTooltipElement();
    const pred = item.prediction;
    const pct = Math.round(pred.prob_clickbait * 100);

    let barColor = "#22c55e";
    let statusDot = "🟢";
    if (pred.badge_level === "red") {
      barColor = "#ef4444";
      statusDot = "🔴";
    } else if (pred.badge_level === "yellow") {
      barColor = "#f59e0b";
      statusDot = "🟡";
    }

    const triggers = pred.trigger_words || [];
    const chipsHtml = triggers.length
      ? triggers.map(t => `<span class="vn-acb-tt-chip">⚡ ${escapeHtml(t.word)}</span>`).join("")
      : `<span style="color:#94a3b8;font-size:12px;">Không có từ khóa kích động mạnh</span>`;

    const reasonsHtml = (pred.reasons_vi || [])
      .slice(0, 2)
      .map(r => `<div style="margin-bottom:3px;">• ${escapeHtml(r)}</div>`)
      .join("");

    const showDeclickbaitBtn =
      (pred.is_clickbait || pred.badge_level === "yellow") &&
      pred.declickbait_title &&
      pred.declickbait_title !== item.originalTitle;

    tt.innerHTML = `
      <div class="vn-acb-tt-header">
        <div class="vn-acb-tt-title">
          <span>${statusDot}</span>
          <span>${escapeHtml(pred.verdict_vi)}</span>
        </div>
        <span style="font-weight:800;color:${barColor};font-size:13px;">${pct}%</span>
      </div>
      <div class="vn-acb-tt-meter">
        <div class="vn-acb-tt-meter-fill" style="width:${Math.max(pct, 4)}%;background:${barColor};"></div>
      </div>
      <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">
        ${reasonsHtml}
      </div>
      <div>${chipsHtml}</div>
      <div class="vn-acb-tt-section" id="vn-acb-tt-sapo-box">
        <div style="font-weight:700;color:#38bdf8;font-size:11.5px;margin-bottom:2px;">
          💡 Tóm tắt nội dung thật (Sapo):
        </div>
        <div id="vn-acb-tt-sapo-text">${escapeHtml(pred.quick_summary)}</div>
      </div>
      ${
        showDeclickbaitBtn
          ? `<div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;gap:6px;">
               <button class="vn-acb-tt-btn" id="vn-acb-tt-neutral-btn">✨ Tháo tít trên trang</button>
               <button class="vn-acb-tt-btn" style="background:#334155;" id="vn-acb-tt-report-btn">🚩 Báo nhầm</button>
             </div>`
          : ""
      }
    `;

    tt.style.display = "block";
    positionTooltip(tt, anchorEl);
    requestAnimationFrame(() => tt.classList.add("vn-acb-visible"));

    // Gắn sự kiện nút Tháo tít trực tiếp trên trang
    const neutralBtn = tt.querySelector("#vn-acb-tt-neutral-btn");
    if (neutralBtn) {
      neutralBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (item.titleEl) {
          item.titleEl.textContent = `✨ ${pred.declickbait_title}`;
          item.titleEl.classList.remove("vn-acb-blurred-title");
          neutralBtn.textContent = "✓ Đã đổi sang tít trung lập";
        }
      });
    }

    const reportBtn = tt.querySelector("#vn-acb-tt-report-btn");
    if (reportBtn) {
      reportBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        chrome.runtime.sendMessage({
          type: "SUBMIT_FEEDBACK",
          payload: {
            url: item.linkUrl || window.location.href,
            title: item.originalTitle,
            lead_paragraph: item.leadText || "",
            predicted_label: pred.label,
            user_suggested_label: pred.is_clickbait ? "non-clickbait" : "clickbait"
          }
        });
        reportBtn.textContent = "✓ Đã gửi phản hồi";
      });
    }

    // Nếu chưa có Sapo trên trang danh sách nhưng có URL bài báo, tự động tải Sapo ngầm
    if (
      (!item.leadText || item.leadText.length < 15) &&
      item.linkUrl &&
      item.linkUrl.startsWith("http") &&
      !item.sapoFetched
    ) {
      item.sapoFetched = true;
      const sapoEl = tt.querySelector("#vn-acb-tt-sapo-text");
      if (sapoEl) sapoEl.textContent = "⏳ Đang trích xuất nhanh nội dung Sapo từ bài viết...";
      chrome.runtime.sendMessage(
        {
          type: "FETCH_ARTICLE_SAPO",
          url: item.linkUrl,
          title: item.originalTitle
        },
        (resp) => {
          if (resp && resp.ok && resp.result) {
            item.prediction = resp.result;
            item.leadText = resp.result.quick_summary;
            const currentSapoEl = document.getElementById("vn-acb-tt-sapo-text");
            if (currentSapoEl) {
              currentSapoEl.textContent = resp.result.quick_summary;
            }
          }
        }
      );
    }
  }

  function positionTooltip(tt, anchorEl) {
    const rect = anchorEl.getBoundingClientRect();
    const ttWidth = 350;
    const ttHeight = tt.offsetHeight || 210;

    let left = rect.left;
    if (left + ttWidth > window.innerWidth - 16) {
      left = window.innerWidth - ttWidth - 16;
    }
    if (left < 12) left = 12;

    let top = rect.bottom + 8;
    if (top + ttHeight > window.innerHeight - 16) {
      top = Math.max(12, rect.top - ttHeight - 8);
    }

    tt.style.left = `${left}px`;
    tt.style.top = `${top}px`;
  }

  function hideTooltip() {
    if (tooltipHideTimer) clearTimeout(tooltipHideTimer);
    tooltipHideTimer = setTimeout(() => {
      if (tooltipEl) {
        tooltipEl.classList.remove("vn-acb-visible");
        setTimeout(() => {
          if (tooltipEl && !tooltipEl.classList.contains("vn-acb-visible")) {
            tooltipEl.style.display = "none";
          }
        }, 180);
      }
    }, 180);
  }

  /**
   * 2. THU THẬP CÁC BÀI BÁO TRÊN TRANG CHỦ / CHUYÊN MỤC (TẦNG 1)
   */
  function collectArticleElements() {
    const selectors = [
      "[data-acb-article]",
      "article.item-news",
      "article.article-item",
      ".box-category-item",
      ".story",
      ".knswli",
      ".news-item",
      "h2.title-news",
      "h3.title-news",
      "h3.article-title",
      "h2.article-title",
      "h3.knswli-title"
    ];

    const candidates = [];
    const seenTitleNodes = new Set();

    // Ưu tiên tìm theo thẻ card bài viết hoặc các thẻ tiêu đề h2/h3 chứa link, hỗ trợ cả thẻ a.title trực tiếp (như bongdaplus.vn)
    const headlineLinks = document.querySelectorAll(
      "[data-acb-title], h1 a, h2 a, h3 a, h4 a, .title-news a, .article-title a, .story__heading a, .knswli-title a, a.title, a.tit, a.post-title, a.article-title, .title a, .tit a"
    );

    headlineLinks.forEach((linkEl) => {
      if (seenTitleNodes.has(linkEl)) return;
      if (linkEl.closest(".vn-acb-reading-pill") || linkEl.closest(".vn-acb-tooltip")) return;

      const text = (linkEl.textContent || "").replace(/\s+/g, " ").trim();
      // Tiêu đề bài báo tiếng Việt thường từ 6 từ trở lên (>= 28 ký tự)
      if (text.length < 26 || text.split(/\s+/).length < 6) return;

      seenTitleNodes.add(linkEl);

      // Tìm thẻ bao ngoài (card) để lấy đoạn sapo đi kèm nếu có
      const card =
        linkEl.closest("[data-acb-article], article, .item-news, .box-category-item, .knswli, .news-item, .story, .news, .item, li, .card") ||
        linkEl.parentElement?.parentElement;

      let leadText = "";
      if (card) {
        const sapoAttr = card.getAttribute("data-acb-sapo");
        if (sapoAttr) {
          leadText = sapoAttr.trim();
        } else {
          const descEl = card.querySelector(
            "[data-acb-sapo], .description, .sapo, .summary, .article-excerpt, .knswli-sapo, .story__summary, p"
          );
          if (descEl && descEl !== linkEl && !descEl.contains(linkEl)) {
            leadText = (descEl.textContent || "").replace(/\s+/g, " ").trim();
          }
        }
      }

      candidates.push({
        titleEl: linkEl,
        containerEl: linkEl.parentElement || linkEl,
        originalTitle: text,
        leadText,
        linkUrl: linkEl.href || ""
      });
    });

    return candidates.slice(0, 60); // Giới hạn 60 tiêu đề đầu trang để đảm bảo tốc độ <20ms
  }

  /**
   * 3. QUÉT VÀ GẮN NHÃN TÍN HIỆU (TẦNG 1)
   */
  function scanAndAnnotatePage() {
    // Xóa các badge cũ nếu quét lại
    document.querySelectorAll(".vn-acb-badge").forEach(el => el.remove());
    document.querySelectorAll(".vn-acb-blurred-title").forEach(el => el.classList.remove("vn-acb-blurred-title"));
    pageArticlesMap.clear();

    if (!currentSettings.enabled) return;
    const domain = window.location.hostname.replace(/^www\./, "");
    if ((currentSettings.whitelistedDomains || []).includes(domain)) return;

    const articles = collectArticleElements();
    if (articles.length === 0) return;

    const batchPayload = [];
    articles.forEach((art, idx) => {
      const id = `acb_art_${idx}`;
      pageArticlesMap.set(id, { ...art, id, prediction: null, badgeEl: null });
      batchPayload.push({
        id,
        title: art.originalTitle,
        lead_paragraph: art.leadText || ""
      });
    });

    chrome.runtime.sendMessage(
      {
        type: "PREDICT_BATCH",
        items: batchPayload
      },
      (response) => {
        if (!response || !response.ok || !response.results) return;

        let cbCount = 0;
        let yellowCount = 0;
        let cleanCount = 0;

        response.results.forEach((pred) => {
          const item = pageArticlesMap.get(pred.id);
          if (!item) return;
          item.prediction = pred;

          if (pred.badge_level === "red") cbCount++;
          else if (pred.badge_level === "yellow") yellowCount++;
          else cleanCount++;

          renderBadgeForItem(item, pred);
        });

        pageStats = {
          domain: domain || "demo-news",
          totalScanned: response.results.length,
          clickbaitCount: cbCount,
          yellowCount,
          cleanCount,
          engineMode: response.engine || "local_ml"
        };
      }
    );
  }

  function renderBadgeForItem(item, pred) {
    const badge = document.createElement("span");
    let levelClass = "vn-acb-badge-green";
    let dot = "🟢";

    if (pred.badge_level === "red") {
      levelClass = "vn-acb-badge-red";
      dot = "🔴";
    } else if (pred.badge_level === "yellow") {
      levelClass = "vn-acb-badge-yellow";
      dot = "🟡";
    }

    badge.className = `vn-acb-badge ${levelClass}`;
    badge.innerHTML = `<span>${dot}</span><span>${escapeHtml(pred.badge_text)}</span>`;

    // Áp dụng chế độ Làm mờ (Blur) nếu người dùng bật
    if (currentSettings.blurClickbait && pred.is_clickbait) {
      item.titleEl.classList.add("vn-acb-blurred-title");
    }

    badge.addEventListener("mouseenter", () => {
      showTooltipForArticle(item.id, badge);
    });
    badge.addEventListener("mouseleave", () => {
      hideTooltip();
    });
    badge.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      showTooltipForArticle(item.id, badge);
    });

    item.titleEl.insertAdjacentElement("afterend", badge);
    item.badgeEl = badge;
  }

  /**
   * 4. TẦNG 2: TRỢ LÝ ĐỌC BÀI CHI TIẾT (READING ASSISTANT FLOATING PILL)
   */
  function initReadingAssistantIfDetailPage() {
    if (readingPillEl) {
      readingPillEl.remove();
      readingPillEl = null;
    }
    if (!currentSettings.enabled || !currentSettings.showReadingAssistant) return;

    const h1El = document.querySelector(
      "[data-acb-detail-title], h1.title-detail, h1.article-title, h1.the-article-title, h1.kbwc-title, h1.detail-title, h1"
    );
    const sapoEl = document.querySelector(
      "[data-acb-detail-sapo], p.description, h2.sapo, div.sapo, h2.detail-sapo, .singular-sapo, .article-sapo, .summary, .cms-desc"
    );

    if (!h1El) return;
    const titleText = (h1El.textContent || "").replace(/\s+/g, " ").trim();
    if (titleText.length < 20) return;
    const leadText = sapoEl ? (sapoEl.textContent || "").replace(/\s+/g, " ").trim() : "";

    chrome.runtime.sendMessage(
      {
        type: "PREDICT_SINGLE",
        title: titleText,
        lead_paragraph: leadText
      },
      (resp) => {
        if (!resp || !resp.ok || !resp.result) return;
        renderReadingPill(h1El, titleText, leadText, resp.result);
      }
    );
  }

  function renderReadingPill(h1El, originalTitle, leadText, pred) {
    if (readingPillEl) readingPillEl.remove();

    const pill = document.createElement("div");
    pill.className = "vn-acb-reading-pill";

    const pct = Math.round(pred.prob_clickbait * 100);
    let dot = "🟢";
    let accent = "#4ade80";
    if (pred.badge_level === "red") {
      dot = "🔴";
      accent = "#f87171";
    } else if (pred.badge_level === "yellow") {
      dot = "🟡";
      accent = "#fbbf24";
    }

    const reasonText = (pred.reasons_vi && pred.reasons_vi[0]) || "";
    let isReplaced = false;

    pill.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <div style="font-weight:800;font-size:13px;display:flex;align-items:center;gap:6px;">
          <span>🛡️ Trợ lý Đọc Báo</span>
          <span style="color:${accent};">· ${dot} ${escapeHtml(pred.badge_text)}</span>
        </div>
        <button id="vn-acb-pill-min-btn" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:14px;font-weight:800;" title="Thu gọn">—</button>
      </div>
      <div id="vn-acb-pill-body">
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">
          ${escapeHtml(reasonText)}
        </div>
        ${
          pred.is_clickbait && pred.declickbait_title
            ? `<div style="background:#1e293b;padding:8px 10px;border-radius:8px;border-left:3px solid #38bdf8;font-size:12px;margin-bottom:8px;">
                 <div style="color:#38bdf8;font-weight:700;font-size:11px;">✨ Tiêu đề trung lập sát nội dung:</div>
                 <div style="color:#f8fafc;font-weight:600;margin-top:2px;">“${escapeHtml(pred.declickbait_title)}”</div>
               </div>`
            : ""
        }
        <div class="vn-acb-pill-actions">
          ${
            pred.is_clickbait
              ? `<button class="vn-acb-pill-btn vn-acb-primary" id="vn-acb-pill-declickbait">✨ Tháo tít trên bài</button>`
              : ""
          }
          <button class="vn-acb-pill-btn" id="vn-acb-pill-feedback">🚩 Báo nhầm lẫn</button>
        </div>
      </div>
    `;

    document.body.appendChild(pill);
    readingPillEl = pill;

    const minBtn = pill.querySelector("#vn-acb-pill-min-btn");
    const bodyEl = pill.querySelector("#vn-acb-pill-body");
    if (minBtn && bodyEl) {
      minBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const minimized = pill.classList.toggle("vn-acb-minimized");
        bodyEl.style.display = minimized ? "none" : "block";
        minBtn.textContent = minimized ? "▲" : "—";
      });
    }

    const declickBtn = pill.querySelector("#vn-acb-pill-declickbait");
    if (declickBtn) {
      declickBtn.addEventListener("click", () => {
        if (!isReplaced) {
          h1El.textContent = pred.declickbait_title;
          declickBtn.textContent = "↩ Khôi phục tít gốc";
          isReplaced = true;
        } else {
          h1El.textContent = originalTitle;
          declickBtn.textContent = "✨ Tháo tít trên bài";
          isReplaced = false;
        }
      });
    }

    const fbBtn = pill.querySelector("#vn-acb-pill-feedback");
    if (fbBtn) {
      fbBtn.addEventListener("click", () => {
        chrome.runtime.sendMessage({
          type: "SUBMIT_FEEDBACK",
          payload: {
            url: window.location.href,
            title: originalTitle,
            lead_paragraph: leadText,
            predicted_label: pred.label,
            user_suggested_label: pred.is_clickbait ? "non-clickbait" : "clickbait"
          }
        });
        fbBtn.textContent = "✓ Cảm ơn phản hồi!";
        fbBtn.disabled = true;
      });
    }
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * 5. LẮNG NGHE LỆNH TỪ POPUP (TẦNG 3) & KHỞI ĐỘNG
   */
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "GET_PAGE_SUMMARY") {
      sendResponse({ ok: true, pageStats });
    } else if (msg.type === "SETTINGS_UPDATED") {
      currentSettings = { ...currentSettings, ...msg.settings };
      scanAndAnnotatePage();
      initReadingAssistantIfDetailPage();
      sendResponse({ ok: true });
    } else if (msg.type === "TRIGGER_ARTICLE_MODAL") {
      // Cho phép trang demo_news.html kích hoạt Reading Assistant khi mở bài chi tiết
      setTimeout(() => initReadingAssistantIfDetailPage(), 150);
      sendResponse({ ok: true });
    }
  });

  // Lắng nghe custom event từ trang demo_news.html khi chuyển đổi chế độ xem bài báo
  window.addEventListener("vn-acb-rescan", () => {
    scanAndAnnotatePage();
    initReadingAssistantIfDetailPage();
  });

  // Khởi chạy lần đầu
  chrome.storage.local.get(null, (stored) => {
    currentSettings = { ...currentSettings, ...stored };
    scanAndAnnotatePage();
    initReadingAssistantIfDetailPage();
  });
})();
