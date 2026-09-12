// --------------------------------------------------------------------
// Navigation entre vues
// --------------------------------------------------------------------
const views = {
  home: document.getElementById("view-home"),
  upload: document.getElementById("view-upload"),
  results: document.getElementById("view-results"),
  history: document.getElementById("view-history"),
  about: document.getElementById("view-about"),
};

const navlinks = document.querySelectorAll(".navlink");

function showView(name) {
  Object.entries(views).forEach(([key, el]) => {
    el.hidden = key !== name;
  });
  navlinks.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: "instant" });
}

document.querySelectorAll("[data-view]").forEach((el) => {
  el.addEventListener("click", () => showView(el.dataset.view));
});

document.getElementById("ctaAnalyze").addEventListener("click", () => showView("upload"));
document.getElementById("reanalyzeBtn").addEventListener("click", () => {
  resetUpload();
  showView("upload");
});

// --------------------------------------------------------------------
// Upload (drag & drop / sélection de fichier)
// --------------------------------------------------------------------
const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

let selectedFile = null;
let selectedFileDataUrl = null;

dropzone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("is-dragover"); });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Le fichier sélectionné n'est pas une image.");
    return;
  }
  selectedFile = file;
  hideError();

  const reader = new FileReader();
  reader.onload = (e) => {
    selectedFileDataUrl = e.target.result;
    previewImage.src = selectedFileDataUrl;
    previewWrap.hidden = false;
    dropzoneInner.hidden = true;
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
}

function resetUpload() {
  selectedFile = null;
  selectedFileDataUrl = null;
  fileInput.value = "";
  previewWrap.hidden = true;
  dropzoneInner.hidden = false;
  analyzeBtn.disabled = true;
  hideError();
}

// --------------------------------------------------------------------
// Analyse
// --------------------------------------------------------------------
const resultOriginal = document.getElementById("resultOriginal");
const heatmapImage = document.getElementById("heatmapImage");
const verdictValue = document.getElementById("verdictValue");
const verdictIcon = document.getElementById("verdictIcon");
const confidenceValue = document.getElementById("confidenceValue");
const benignProb = document.getElementById("benignProb");
const malignantProb = document.getElementById("malignantProb");
const benignBar = document.getElementById("benignBar");
const malignantBar = document.getElementById("malignantBar");
const explanationText = document.getElementById("explanationText");

const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");
const sessionHistory = [];

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("is-loading");
  analyzeBtn.querySelector(".btn__label").textContent = "Analyse en cours…";
  dropzone.classList.add("is-scanning");
  hideError();

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Une erreur est survenue.");

    renderResult(data);
    addToHistory(data);
    showView("results");
  } catch (err) {
    showError(err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("is-loading");
    analyzeBtn.querySelector(".btn__label").textContent = "Analyser cette image";
    dropzone.classList.remove("is-scanning");
  }
});

function renderResult(data) {
  resultOriginal.src = selectedFileDataUrl;
  heatmapImage.src = data.heatmap_image;

  const isMalignant = data.predicted_class === "malignant";

  verdictValue.textContent = data.predicted_class === "malignant" ? "Malin" : "Bénin";
  verdictIcon.classList.toggle("is-malignant", isMalignant);
  confidenceValue.textContent = `Confiance : ${data.confidence}%`;

  benignProb.textContent = `${data.benign_probability}%`;
  malignantProb.textContent = `${data.malignant_probability}%`;
  benignBar.style.width = `${data.benign_probability}%`;
  malignantBar.style.width = `${data.malignant_probability}%`;

  explanationText.textContent = isMalignant
    ? "Le modèle a détecté des caractéristiques davantage associées aux lésions malignes dans la zone mise en évidence ci-contre (Grad-CAM). Cela ne remplace pas un examen clinique."
    : "Le modèle n'a pas détecté de caractéristiques fortement associées à une lésion maligne. La zone mise en évidence ci-contre (Grad-CAM) montre la région ayant le plus influencé cette décision.";
}

function addToHistory(data) {
  sessionHistory.unshift({
    thumbnail: selectedFileDataUrl,
    predictedClass: data.predicted_class,
    confidence: data.confidence,
    time: new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }),
  });
  renderHistory();
}

function renderHistory() {
  if (sessionHistory.length === 0) {
    historyEmpty.hidden = false;
    historyList.innerHTML = "";
    return;
  }
  historyEmpty.hidden = true;
  historyList.innerHTML = sessionHistory.map((item) => `
    <div class="history-item">
      <img src="${item.thumbnail}" alt="">
      <div>
        <div class="history-item__label">
          <span class="history-item__dot history-item__dot--${item.predictedClass === "malignant" ? "malignant" : "benign"}"></span>
          ${item.predictedClass === "malignant" ? "Malin" : "Bénin"}
        </div>
        <div class="history-item__meta">${item.time} · confiance ${item.confidence}%</div>
      </div>
    </div>
  `).join("");
}

function showError(message) {
  errorMessage.textContent = message;
  errorSection.hidden = false;
}

function hideError() {
  errorSection.hidden = true;
}
