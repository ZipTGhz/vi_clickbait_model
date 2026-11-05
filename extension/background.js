/**
 * Background Service Worker (Manifest V3) — VN Anti-Clickbait Shield
 * Đảm nhiệm:
 * 1. Cầu nối giao tiếp với AI Backend Server (FastAPI + Gradio tại http://127.0.0.1:7860) không bị chặn bởi CORS/Mixed-Content.
 * 2. Bộ máy suy diễn Tầng 1 (Client-side Fallback Engine) sử dụng trực tiếp trọng số thực tế từ model_weights.json
 *    để Extension hoạt động mượt mà trong <1ms ngay cả khi không bật server Python.
 * 3. Quản lý thống kê hiệu quả bảo vệ (Số bài giật tít đã cảnh báo & Số phút tiết kiệm được).
 */

const DEFAULT_SETTINGS = {
  enabled: true,
  blurClickbait: false,
  showHoverSummary: true,
  showReadingAssistant: true,
  sensitivity: 0.50,
  apiUrl: "http://127.0.0.1:7860",
  whitelistedDomains: [],
  stats: {
    totalScanned: 0,
    clickbaitDetected: 0,
    minutesSaved: 0
  }
};

let MODEL_WEIGHTS = null;

async function loadLocalWeights() {
  if (MODEL_WEIGHTS) return MODEL_WEIGHTS;
  try {
    const url = chrome.runtime.getURL("model_weights.json");
    const resp = await fetch(url);
    MODEL_WEIGHTS = await resp.json();
  } catch (err) {
    console.warn("[Anti-Clickbait] Không thể nạp model_weights.json, dùng bộ từ khóa mặc định:", err);
    MODEL_WEIGHTS = { token_weights: {}, sensational_phrases: [], orthodox_phrases: [] };
  }
  return MODEL_WEIGHTS;
}

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.local.get(null);
  const merged = { ...DEFAULT_SETTINGS, ...existing };
  if (!existing.stats) {
    merged.stats = DEFAULT_SETTINGS.stats;
  }
  await chrome.storage.local.set(merged);
  await loadLocalWeights();
});

/**
 * Suy diễn Tầng 1 (Local ML Scoring) dựa trên trọng số n-gram từ Linear SVM/Logistic Regression
 * kết hợp 12 đặc trưng thủ công (Hand-crafted features).
 */
async function predictLocalSingle(title, leadParagraph = "", sensitivity = 0.50) {
  const weightsData = await loadLocalWeights();
  const tokenWeights = weightsData.token_weights || {};
  const sensationalPhrases = weightsData.sensational_phrases || [];
  const orthodoxPhrases = weightsData.orthodox_phrases || [];

  const cleanTitle = (title || "").trim();
  const cleanLead = (leadParagraph || "").trim();
  const lowerTitle = cleanTitle.toLowerCase();
  const lowerLead = cleanLead.toLowerCase();

  let rawScore = -0.35; // Bias cơ sở (do tỉ lệ clickbait trong tập dữ liệu ~31.2%)
  const triggerWords = [];
  const seenTriggers = new Set();

  // 1. Quét các cụm từ giật gân đặc trưng (Multi-word phrases)
  for (const phrase of sensationalPhrases) {
    if (lowerTitle.includes(phrase)) {
      const segKey = phrase.replace(/\s+/g, "_");
      const w = tokenWeights[segKey] || tokenWeights[phrase] || 1.15;
      rawScore += Math.max(w * 0.55, 0.45);
      if (!seenTriggers.has(phrase)) {
        seenTriggers.add(phrase);
        triggerWords.push({ word: phrase, weight: Number(w.toFixed(2)) });
      }
    }
  }

  // 2. Quét các cụm từ chính luận / chính thống
  for (const phrase of orthodoxPhrases) {
    if (lowerTitle.includes(phrase) || lowerLead.includes(phrase)) {
      rawScore -= 0.45;
    }
  }

  // 3. Quét unigram & bigram theo trọng số thực tế trong model_weights.json
  const words = lowerTitle
    .replace(/[.,:;()[\]"'“”‘’?!…]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  for (let i = 0; i < words.length; i++) {
    const u = words[i];
    if (i + 1 < words.length) {
      const compound = `${u}_${words[i + 1]}`;
      const cw = tokenWeights[compound];
      if (cw !== undefined) {
        rawScore += cw * 0.28;
        const disp = `${u} ${words[i + 1]}`;
        if (cw >= 0.45 && !seenTriggers.has(disp)) {
          seenTriggers.add(disp);
          triggerWords.push({ word: disp, weight: Number(cw.toFixed(2)) });
        }
      }
    }
    const uw = tokenWeights[u];
    if (uw !== undefined && Math.abs(uw) >= 0.35) {
      rawScore += uw * 0.16;
      if (uw >= 0.65 && u.length > 2 && !seenTriggers.has(u)) {
        seenTriggers.add(u);
        triggerWords.push({ word: u, weight: Number(uw.toFixed(2)) });
      }
    }
  }

  // 4. Cộng trọng số từ các đặc trưng thủ công (Handcrafted features)
  const hasQuestion = /\?/.test(cleanTitle);
  const hasExclamation = /!/.test(cleanTitle);
  const hasQuotes = /["'“”‘’]/.test(cleanTitle);
  const hasEllipsis = /\.\.\.|…/.test(cleanTitle);
  const numDigits = (cleanTitle.match(/\d+/g) || []).length;

  if (hasQuestion) rawScore += 0.55;
  if (hasExclamation) rawScore += 0.65;
  if (hasQuotes) rawScore += 0.25;
  if (hasEllipsis) rawScore += 0.35;
  if (numDigits > 0 && triggerWords.length > 0) rawScore += 0.22;

  // Hàm Sigmoid quy đổi sang xác suất [0, 1]
  const prob = 1 / (1 + Math.exp(-rawScore));
  const isClickbait = prob >= sensitivity;

  let badgeLevel = "green";
  let badgeText = "Không giật tít";
  let verdictVi = "TIN KHÔNG GIẬT TÍT";

  if (prob >= Math.max(sensitivity + 0.12, 0.62)) {
    badgeLevel = "red";
    badgeText = `Giật tít · ${Math.round(prob * 100)}%`;
    verdictVi = "TIN GIẬT TÍT / CÂU VIEW CAO";
  } else if (prob >= Math.max(sensitivity - 0.10, 0.40)) {
    badgeLevel = "yellow";
    badgeText = `Chú ý · ${Math.round(prob * 100)}%`;
    verdictVi = "TIÊU ĐỀ CÓ YẾU TỐ GÂY TÒ MÒ";
  }

  triggerWords.sort((a, b) => b.weight - a.weight);

  // Sinh lý do giải thích ngắn gọn
  const reasonsVi = [];
  if (triggerWords.length > 0) {
    const topStr = triggerWords.slice(0, 3).map(t => `"${t.word}"`).join(", ");
    reasonsVi.push(`Chứa từ khóa kích thích tò mò/cảm xúc mạnh: ${topStr}.`);
  }
  if (hasQuestion || hasExclamation) {
    reasonsVi.push("Sử dụng dấu câu cảm thán (? / !) nhằm thu hút sự chú ý tức thì.");
  }
  if (/thứ này|điều này|người này|cô gái này|chàng trai này|nơi này/.test(lowerTitle)) {
    reasonsVi.push("Cố tình giấu chủ thể bằng đại từ bỏ lửng để buộc người đọc phải bấm vào bài.");
  }
  if (reasonsVi.length === 0) {
    reasonsVi.push(
      isClickbait
        ? "Cấu trúc tiêu đề nhấn mạnh chi tiết kịch tính nhằm câu view."
        : "Tiêu đề cung cấp thông tin trực diện, rõ chủ thể và khách quan."
    );
  }

  // Tóm tắt Sapo & Tiêu đề trung lập (De-clickbait)
  let quickSummary = "Rê chuột vào bài hoặc mở bài viết để xem tóm tắt nội dung thật.";
  if (cleanLead && cleanLead.length >= 15) {
    quickSummary = cleanLead.length > 180 ? cleanLead.slice(0, 177) + "..." : cleanLead;
  }

  let declickbaitTitle = cleanTitle;
  if (isClickbait || badgeLevel === "yellow") {
    if (cleanLead && cleanLead.split(/\s+/).length >= 8) {
      const firstSent = cleanLead.split(/(?<=[.!?])\s+/)[0].replace(/[.!?]+$/, "");
      declickbaitTitle = firstSent.length > 105 ? firstSent.slice(0, 102) + "..." : firstSent;
    } else {
      declickbaitTitle = cleanTitle
        .replace(/bất ngờ|ngỡ ngàng|choáng váng|xôn xao|gây bão|dậy sóng|hé lộ|tiết lộ/gi, "")
        .replace(/và cái kết.*$/i, "")
        .replace(/khiến ai cũng.*$/i, "")
        .replace(/[?!]+$/g, "")
        .replace(/\s+/g, " ")
        .trim();
    }
  }

  return {
    label: isClickbait ? "clickbait" : "non-clickbait",
    is_clickbait: isClickbait,
    prob_clickbait: Number(prob.toFixed(4)),
    badge_level: badgeLevel,
    badge_text: badgeText,
    verdict_vi: verdictVi,
    trigger_words: triggerWords.slice(0, 5),
    reasons_vi: reasonsVi,
    quick_summary: quickSummary,
    declickbait_title: declickbaitTitle,
    engine_mode: "local_ml"
  };
}

/**
 * Thử gọi API Backend (Ensemble 4 mô hình) với timeout ngắn, nếu tắt server thì tự động chạy Local ML.
 */
async function handlePredictBatch(items, sensitivity, apiUrl) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2200);
    const resp = await fetch(`${apiUrl}/api/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items, sensitivity }),
      signal: controller.signal
    });
    clearTimeout(timer);
    if (resp.ok) {
      const data = await resp.json();
      return { results: data.results, engine: "server_ensemble" };
    }
  } catch (_) {
    // Fallback xuống Local ML Engine
  }

  const results = [];
  for (const item of items) {
    const pred = await predictLocalSingle(item.title, item.lead_paragraph || "", sensitivity);
    results.push({ id: item.id, ...pred });
  }
  return { results, engine: "local_ml" };
}

async function handlePredictSingle(title, leadParagraph, sensitivity, apiUrl) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    const resp = await fetch(`${apiUrl}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        lead_paragraph: leadParagraph || "",
        model_choice: "Ensemble (Khuyên dùng)",
        sensitivity
      }),
      signal: controller.signal
    });
    clearTimeout(timer);
    if (resp.ok) {
      const data = await resp.json();
      return { ...data, engine_mode: "server_ensemble" };
    }
  } catch (_) {
    // Fallback
  }
  return await predictLocalSingle(title, leadParagraph, sensitivity);
}

/**
 * Trích xuất nhanh đoạn Sapo từ URL bài báo khi người dùng rê chuột vào tiêu đề trên trang chủ.
 */
async function handleFetchArticleSapo(articleUrl, title, sensitivity, apiUrl) {
  // Ưu tiên nhờ Backend trích xuất nếu đang chạy
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    const resp = await fetch(`${apiUrl}/api/analyze_url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: articleUrl, sensitivity }),
      signal: controller.signal
    });
    clearTimeout(timer);
    if (resp.ok) {
      return await resp.json();
    }
  } catch (_) {}

  // Nếu chạy thuần trên trình duyệt: Fetch HTML bài viết để lấy thẻ meta og:description
  try {
    const resp = await fetch(articleUrl, { credentials: "omit" });
    const html = await resp.text();
    const ogDescMatch =
      html.match(/<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']+)["']/i) ||
      html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']/i);
    const lead = ogDescMatch ? ogDescMatch[1].trim() : "";
    if (lead) {
      return await handlePredictSingle(title, lead, sensitivity, apiUrl);
    }
  } catch (_) {}

  return null;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    const settings = await chrome.storage.local.get(DEFAULT_SETTINGS);
    const apiUrl = settings.apiUrl || "http://127.0.0.1:7860";
    const sensitivity = Number(settings.sensitivity || 0.50);

    if (message.type === "PREDICT_BATCH") {
      const data = await handlePredictBatch(message.items || [], sensitivity, apiUrl);

      // Cập nhật thống kê bảo vệ người dùng
      const cbCount = data.results.filter(r => r.is_clickbait).length;
      const curStats = settings.stats || { totalScanned: 0, clickbaitDetected: 0, minutesSaved: 0 };
      const updatedStats = {
        totalScanned: curStats.totalScanned + data.results.length,
        clickbaitDetected: curStats.clickbaitDetected + cbCount,
        minutesSaved: Number((curStats.minutesSaved + cbCount * 1.5).toFixed(1))
      };
      await chrome.storage.local.set({ stats: updatedStats });

      if (sender.tab && sender.tab.id) {
        chrome.action.setBadgeText({
          tabId: sender.tab.id,
          text: cbCount > 0 ? String(cbCount) : ""
        });
        chrome.action.setBadgeBackgroundColor({
          tabId: sender.tab.id,
          color: cbCount > 0 ? "#dc2626" : "#16a34a"
        });
      }

      sendResponse({ ok: true, ...data, stats: updatedStats });
    } else if (message.type === "PREDICT_SINGLE") {
      const res = await handlePredictSingle(
        message.title || "",
        message.lead_paragraph || "",
        sensitivity,
        apiUrl
      );
      sendResponse({ ok: true, result: res });
    } else if (message.type === "FETCH_ARTICLE_SAPO") {
      const res = await handleFetchArticleSapo(
        message.url,
        message.title || "",
        sensitivity,
        apiUrl
      );
      sendResponse({ ok: Boolean(res), result: res });
    } else if (message.type === "SUBMIT_FEEDBACK") {
      try {
        await fetch(`${apiUrl}/api/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(message.payload)
        });
      } catch (_) {}
      sendResponse({ ok: true });
    } else if (message.type === "CHECK_BACKEND") {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1200);
        const resp = await fetch(`${apiUrl}/api/health`, { signal: controller.signal });
        clearTimeout(timer);
        sendResponse({ online: resp.ok });
      } catch (_) {
        sendResponse({ online: false });
      }
    }
  })();
  return true; // Giữ kênh message mở cho async response
});
