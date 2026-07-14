/* ── app.js — FitBuddy AI frontend logic ──────────────────────── */

const form       = document.getElementById("profileForm");
const submitBtn  = document.getElementById("submitBtn");
const btnText    = submitBtn.querySelector(".btn-text");
const btnLoader  = submitBtn.querySelector(".btn-loader");
const errorBanner= document.getElementById("errorBanner");
const errorMsg   = document.getElementById("errorMsg");
const resultCard = document.getElementById("resultCard");
const planOutput = document.getElementById("planOutput");
const copyBtn    = document.getElementById("copyBtn");

// ── Validation ───────────────────────────────────────────────────
function validateForm(data) {
  const required = ["age", "location", "lifestyle", "weight", "fitness_level"];
  const errors = [];
  required.forEach(f => {
    const el = document.getElementById(f);
    if (!data[f]) {
      el.classList.add("invalid");
      errors.push(f);
    } else {
      el.classList.remove("invalid");
    }
  });
  return errors;
}

// ── Set loading state ────────────────────────────────────────────
function setLoading(loading) {
  submitBtn.disabled = loading;
  btnText.hidden     = loading;
  btnLoader.hidden   = !loading;
}

// ── Show / hide error ────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
  resultCard.hidden    = true;
  errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
}
function clearError() { errorBanner.hidden = true; }

// ── Render plan with highlighted sections ────────────────────────
function renderPlan(text) {
  // Escape HTML entities
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Highlight CRITICAL MEDICAL ALERT lines
  let html = escaped.replace(
    /(⚠️\s*CRITICAL MEDICAL ALERT[^\n]*)/g,
    '<span class="alert-line">$1</span>'
  );

  // Highlight section headings: **…**
  html = html.replace(
    /\*\*([^*]+)\*\*/g,
    '<span class="section-heading">$1</span>'
  );

  planOutput.innerHTML = html;
}

// ── Copy to clipboard ────────────────────────────────────────────
copyBtn.addEventListener("click", () => {
  const text = planOutput.innerText || planOutput.textContent;
  navigator.clipboard.writeText(text).then(() => {
    copyBtn.textContent = "✅ Copied!";
    setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 2500);
  }).catch(() => {
    copyBtn.textContent = "❌ Failed";
    setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 2500);
  });
});

// ── Form submit ──────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const payload = {
    age:               document.getElementById("age").value.trim(),
    location:          document.getElementById("location").value.trim(),
    lifestyle:         document.getElementById("lifestyle").value.trim(),
    weight:            document.getElementById("weight").value.trim(),
    fitness_level:     document.getElementById("fitness_level").value.trim(),
    health_conditions: document.getElementById("health_conditions").value.trim() || "None",
    allergies:         document.getElementById("allergies").value.trim() || "None",
  };

  const errors = validateForm(payload);
  if (errors.length > 0) {
    showError("Please fill in all required fields: " + errors.join(", ") + ".");
    return;
  }

  setLoading(true);
  resultCard.hidden = true;

  try {
    const res  = await fetch("/generate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const json = await res.json();

    if (!res.ok || json.error) {
      showError(json.error || `Server error (${res.status}). Please try again.`);
      return;
    }

    renderPlan(json.plan);
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError("Network error — could not reach the server. " + err.message);
  } finally {
    setLoading(false);
  }
});

// ── Remove invalid class on input ────────────────────────────────
["age","location","lifestyle","weight","fitness_level"].forEach(id => {
  document.getElementById(id).addEventListener("input", function() {
    if (this.value.trim()) this.classList.remove("invalid");
  });
});

// ── Chatbot widget ───────────────────────────────────────────────
(function () {
  const STORAGE_KEY   = "fitbuddy_chat_session";
  const TTL_MS        = 5 * 60 * 60 * 1000; // 5 hours in milliseconds

  const toggleBtn     = document.getElementById("chatToggleBtn");
  const widget        = document.getElementById("chatWidget");
  const closeBtn      = document.getElementById("chatCloseBtn");
  const clearBtn      = document.getElementById("clearChatBtn");
  const chatForm      = document.getElementById("chatForm");
  const chatInput     = document.getElementById("chatInput");
  const chatMessages  = document.getElementById("chatMessages");
  const sendBtn       = document.getElementById("chatSendBtn");
  const sendText      = sendBtn.querySelector(".chat-send-text");
  const sendLoader    = sendBtn.querySelector(".chat-send-loader");
  const expiryNotice  = document.getElementById("chatExpiryNotice");

  // ── Session persistence (localStorage) ────────────────────────
  function loadSession() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const s = JSON.parse(raw);
      if (!s.sessionId || !s.expiresAt) return null;
      if (Date.now() > s.expiresAt) {
        localStorage.removeItem(STORAGE_KEY);
        return null;
      }
      return s;
    } catch { return null; }
  }

  function saveSession(sessionId, expiresAt) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId, expiresAt }));
  }

  function clearSession() {
    localStorage.removeItem(STORAGE_KEY);
    currentSessionId  = null;
    sessionExpiresAt  = null;
  }

  let storedSession   = loadSession();
  let currentSessionId = storedSession ? storedSession.sessionId : null;
  let sessionExpiresAt = storedSession ? storedSession.expiresAt : null;

  // ── Expiry notice timer ────────────────────────────────────────
  function updateExpiryNotice() {
    if (!sessionExpiresAt) { expiryNotice.textContent = ""; return; }
    const remaining = sessionExpiresAt - Date.now();
    if (remaining <= 0) {
      expiryNotice.textContent = "Session expired. Starting fresh.";
      clearSession();
      return;
    }
    const h = Math.floor(remaining / 3600000);
    const m = Math.floor((remaining % 3600000) / 60000);
    expiryNotice.textContent = `History kept for ${h}h ${m}m`;
  }

  // Update every minute while widget is open
  let expiryInterval = null;
  function startExpiryTimer() {
    updateExpiryNotice();
    expiryInterval = setInterval(updateExpiryNotice, 60000);
  }
  function stopExpiryTimer() {
    clearInterval(expiryInterval);
    expiryInterval = null;
  }

  // ── Toggle open/close ──────────────────────────────────────────
  toggleBtn.addEventListener("click", () => {
    const isHidden = widget.hidden;
    widget.hidden = !isHidden;
    toggleBtn.textContent = isHidden ? "✕" : "💬";
    if (isHidden) {
      startExpiryTimer();
      chatInput.focus();
      scrollToBottom();
    } else {
      stopExpiryTimer();
    }
  });

  closeBtn.addEventListener("click", () => {
    widget.hidden = true;
    toggleBtn.textContent = "💬";
    stopExpiryTimer();
  });

  // ── Clear history ──────────────────────────────────────────────
  clearBtn.addEventListener("click", () => {
    clearSession();
    chatMessages.innerHTML = "";
    appendBotMessage("Chat history cleared. Ask me anything!");
    expiryNotice.textContent = "";
  });

  // ── Helpers ────────────────────────────────────────────────────
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendMessage(role, text) {
    const row    = document.createElement("div");
    row.className = `chat-msg ${role}`;
    const bubble = document.createElement("span");
    bubble.className = "chat-bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendBotMessage(text) { appendMessage("bot", text); }
  function appendUserMessage(text) { appendMessage("user", text); }
  function appendErrorMessage(text) { appendMessage("error", text); }

  // Typing indicator
  function showTyping() {
    const row    = document.createElement("div");
    row.className = "chat-msg bot";
    row.id = "chatTyping";
    const bubble = document.createElement("span");
    bubble.className = "chat-bubble";
    bubble.textContent = "…";
    bubble.style.opacity = ".5";
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
  }
  function hideTyping() {
    const el = document.getElementById("chatTyping");
    if (el) el.remove();
  }

  function setSendLoading(on) {
    sendBtn.disabled   = on;
    sendText.hidden    = on;
    sendLoader.hidden  = !on;
  }

  // ── Send message ───────────────────────────────────────────────
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    appendUserMessage(text);
    setSendLoading(true);
    showTyping();

    const body = { message: text };
    if (currentSessionId) body.session_id = currentSessionId;

    try {
      const res  = await fetch("/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
      });
      const json = await res.json();
      hideTyping();

      if (!res.ok || json.error) {
        appendErrorMessage("Error: " + (json.error || `Server error (${res.status})`));
      } else {
        // Persist session_id and reset 5-hour TTL on every message
        currentSessionId = json.session_id;
        sessionExpiresAt = Date.now() + TTL_MS;
        saveSession(currentSessionId, sessionExpiresAt);
        updateExpiryNotice();
        appendBotMessage(json.reply);
      }
    } catch (err) {
      hideTyping();
      appendErrorMessage("Network error — " + err.message);
    } finally {
      setSendLoading(false);
      chatInput.focus();
    }
  });
})();
