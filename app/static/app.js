const messages = document.getElementById("messages");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const sendButton = document.getElementById("send");
const clearButton = document.getElementById("clear");
const topKInput = document.getElementById("topK");
const statusEl = document.getElementById("status");

const sanitize = (text) => String(text || "").trim();

const scrollToBottom = () => {
  messages.scrollTop = messages.scrollHeight;
};

const setStatus = (state, label) => {
  statusEl.classList.remove("online", "offline");
  if (state) {
    statusEl.classList.add(state);
  }
  statusEl.querySelector(".label").textContent = label;
};

const addMessage = (role, text, sources, pending = false) => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}` + (pending ? " pending" : "");

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;

  const textEl = document.createElement("div");
  textEl.className = "text";
  textEl.textContent = text;

  bubble.appendChild(roleEl);
  bubble.appendChild(textEl);

  if (sources && sources.length) {
    const sourcesEl = buildSources(sources);
    bubble.appendChild(sourcesEl);
  }

  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, textEl, bubble };
};

const buildSources = (sources) => {
  const sourcesEl = document.createElement("div");
  sourcesEl.className = "sources";

  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "Sources";

  const list = document.createElement("ul");

  sources.slice(0, 4).forEach((source) => {
    const item = document.createElement("li");
    item.className = "source-item";

    const meta = document.createElement("div");
    meta.className = "source-meta";

    const id = document.createElement("span");
    id.textContent = source.id || "unknown";
    meta.appendChild(id);

    if (typeof source.score === "number") {
      const score = document.createElement("span");
      score.textContent = `score:${source.score.toFixed(2)}`;
      meta.appendChild(score);
    }

    const metaParts = [];
    if (source.metadata) {
      if (source.metadata.source) {
        metaParts.push(source.metadata.source);
      }
      if (source.metadata.page) {
        metaParts.push(`page:${source.metadata.page}`);
      }
    }
    if (metaParts.length) {
      const metaText = document.createElement("span");
      metaText.textContent = metaParts.join(" | ");
      meta.appendChild(metaText);
    }

    const snippet = document.createElement("div");
    snippet.className = "source-text";
    snippet.textContent = sanitize(source.text).slice(0, 160);

    item.appendChild(meta);
    item.appendChild(snippet);
    list.appendChild(item);
  });

  sourcesEl.appendChild(title);
  sourcesEl.appendChild(list);
  return sourcesEl;
};

const updateMessage = (target, text, sources) => {
  target.wrapper.classList.remove("pending");
  target.textEl.textContent = text;
  const existingSources = target.bubble.querySelector(".sources");
  if (existingSources) {
    existingSources.remove();
  }
  if (sources && sources.length) {
    target.bubble.appendChild(buildSources(sources));
  }
  scrollToBottom();
};

const fetchAnswer = async (query) => {
  const payload = { query };
  const topK = Number.parseInt(topKInput.value, 10);
  if (Number.isFinite(topK) && topK > 0) {
    payload.top_k = topK;
  }

  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }

  return response.json();
};

const handleSubmit = async (event) => {
  event.preventDefault();
  const message = sanitize(promptInput.value);
  if (!message) {
    return;
  }

  promptInput.value = "";
  sendButton.disabled = true;

  addMessage("user", message);
  const pending = addMessage("assistant", "Thinking", [], true);

  try {
    const data = await fetchAnswer(message);
    updateMessage(pending, data.answer || "No answer.", data.sources || []);
  } catch (error) {
    updateMessage(pending, `Error: ${sanitize(error.message)}`, []);
  } finally {
    sendButton.disabled = false;
    promptInput.focus();
  }
};

const init = async () => {
  setStatus("", "checking");
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("health check failed");
    }
    setStatus("online", "online");
  } catch (error) {
    setStatus("offline", "offline");
  }
};

composer.addEventListener("submit", handleSubmit);

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage(
    "assistant",
    'Drop a question to start. Example: "What is RAG?"'
  );
});

init();
