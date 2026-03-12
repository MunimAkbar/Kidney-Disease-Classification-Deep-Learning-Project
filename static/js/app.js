/* ============================================
   Kidney Disease Classifier — App Logic
   ============================================ */

const CONFIDENCE_THRESHOLD = 55; // below this % → "image may be incorrect"
let base64ImageData = "";

// ── Init ────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initUpload();
    initButtons();
});

// ── Upload handling ─────────────────────────
function initUpload() {
    const zone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");

    // Click to browse
    zone.addEventListener("click", () => fileInput.click());

    // Drag events
    ["dragenter", "dragover"].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.remove("dragover");
        });
    });

    zone.addEventListener("drop", e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) processFile(files[0]);
    });

    fileInput.addEventListener("change", e => {
        if (e.target.files.length > 0) processFile(e.target.files[0]);
    });
}

function processFile(file) {
    if (!file.type.match("image.*")) {
        showToast("Please select an image file.", "error");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast("File must be under 10 MB.", "error");
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        const dataUrl = e.target.result;
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = function () {
            // Convert to base64 for the API
            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            canvas.getContext("2d").drawImage(img, 0, 0);
            base64ImageData = canvas
                .toDataURL("image/jpeg", 0.9)
                .replace(/^data:image.+;base64,/, "");

            // Show preview
            const preview = document.getElementById("preview-image");
            const placeholder = document.getElementById("preview-placeholder");
            preview.src = dataUrl;
            preview.style.display = "block";
            placeholder.style.display = "none";

            // Enable analyze button
            document.getElementById("btn-analyze").disabled = false;

            showToast("Image loaded successfully!", "success");
        };
        img.src = dataUrl;
    };
    reader.readAsDataURL(file);
}

// ── Buttons ─────────────────────────────────
function initButtons() {
    document.getElementById("btn-browse").addEventListener("click", () => {
        document.getElementById("file-input").click();
    });

    document.getElementById("btn-analyze").addEventListener("click", () => {
        if (!base64ImageData) {
            showToast("Please upload an image first.", "warning");
            return;
        }
        analyzeImage();
    });
}

// ── API call ────────────────────────────────
function analyzeImage() {
    const overlay = document.getElementById("loading-overlay");
    overlay.classList.add("active");

    // Reset results
    document.getElementById("results-content").innerHTML =
        '<div class="results-placeholder"><div class="icon-big">⏳</div><p>Analyzing…</p></div>';

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64ImageData }),
    })
        .then(res => res.json())
        .then(data => {
            overlay.classList.remove("active");
            if (data.error) {
                showToast("Error: " + data.error, "error");
                return;
            }
            renderResults(data);
            showToast("Analysis complete!", "success");
        })
        .catch(err => {
            overlay.classList.remove("active");
            showToast("Server error. Is the Flask app running?", "error");
            console.error(err);
        });
}

// ── Render results ──────────────────────────
function renderResults(data) {
    const container = document.getElementById("results-content");
    const prediction = data.prediction;   // "Tumor" or "Normal"
    const confidence = data.confidence;   // e.g. 92.34

    const isLowConfidence = confidence < CONFIDENCE_THRESHOLD;

    // Decide badge type
    let badgeClass, badgeIcon, badgeLabel, badgeSub;
    if (isLowConfidence) {
        badgeClass = "invalid";
        badgeIcon = "⚠️";
        badgeLabel = "Uncertain";
        badgeSub = "The uploaded image may not be a valid kidney CT scan.";
    } else if (prediction === "Tumor") {
        badgeClass = "tumor";
        badgeIcon = "🔴";
        badgeLabel = "Tumor Detected";
        badgeSub = "The model indicates abnormal tissue growth.";
    } else {
        badgeClass = "normal";
        badgeIcon = "🟢";
        badgeLabel = "Normal";
        badgeSub = "No abnormalities detected in the scan.";
    }

    // Confidence bar class
    let barClass = "high";
    if (confidence < 60) barClass = "low";
    else if (confidence < 80) barClass = "medium";

    container.innerHTML = `
        <div class="result-display">
            <!-- Badge -->
            <div class="result-badge ${badgeClass}">
                <div class="badge-icon">${badgeIcon}</div>
                <div class="badge-label">${badgeLabel}</div>
                <div class="badge-subtitle">${badgeSub}</div>
            </div>

            <!-- Confidence -->
            <div class="confidence-section">
                <div class="confidence-header">
                    <span>Confidence</span>
                    <span class="conf-value">${confidence}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill ${barClass}" id="conf-bar"></div>
                </div>
            </div>

            <!-- Details -->
            <div class="result-details">
                <div class="detail-row">
                    <span class="label">Classification</span>
                    <span class="value">${prediction}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Confidence Score</span>
                    <span class="value">${confidence}%</span>
                </div>
                <div class="detail-row">
                    <span class="label">Model</span>
                    <span class="value">VGG-16</span>
                </div>
                <div class="detail-row">
                    <span class="label">Image Quality</span>
                    <span class="value">${isLowConfidence ? "Low / Invalid" : "Good"}</span>
                </div>
            </div>
        </div>
    `;

    // Animate confidence bar after a tiny delay
    setTimeout(() => {
        const bar = document.getElementById("conf-bar");
        if (bar) bar.style.width = confidence + "%";
    }, 100);
}

// ── Toast notifications ─────────────────────
function showToast(message, type = "info") {
    let container = document.getElementById("toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.className = "toast-container";
        document.body.appendChild(container);
    }

    const icons = { success: "✅", error: "❌", warning: "⚠️", info: "ℹ️" };
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || "ℹ️"}</span> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(40px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
