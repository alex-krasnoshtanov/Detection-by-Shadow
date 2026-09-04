"use strict";

// The interesting bit of drawing here: the predicted box is mostly OUTSIDE the
// uploaded image, so the canvas cannot just be the image. It is sized to the
// union of the image and the box, with the image drawn at an offset inside it.

const PAD = 24;              // breathing room around the union, in image pixels
const FRAME_STROKE = "#9aa3ad";
const PRED_STROKE = "#ef4444";

const dropzone = document.getElementById("drop");
const fileInput = document.getElementById("file");
const browseButton = document.getElementById("browse");
const statusLine = document.getElementById("status");
const resultBlock = document.getElementById("result");
const statsList = document.getElementById("stats");
const errorBlock = document.getElementById("error");
const canvas = document.getElementById("canvas");

let busy = false;

// ---- model status ---------------------------------------------------------

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const body = await response.json();
    if (body.ready) {
      statusLine.textContent = `model ready on ${body.device}`;
      statusLine.className = "status ready";
    } else {
      statusLine.textContent = "model unavailable";
      statusLine.className = "status down";
      showError(body.error || "the server did not load a model");
    }
  } catch (error) {
    statusLine.textContent = "server unreachable";
    statusLine.className = "status down";
  }
}

// ---- upload ---------------------------------------------------------------

browseButton.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) submit(fileInput.files[0]);
});

for (const name of ["dragenter", "dragover"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  });
}
for (const name of ["dragleave", "drop"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
  });
}
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) submit(file);
});

// Paste a screenshot straight in.
window.addEventListener("paste", (event) => {
  for (const item of event.clipboardData?.items ?? []) {
    if (item.type.startsWith("image/")) {
      submit(item.getAsFile());
      return;
    }
  }
});

async function submit(file) {
  if (busy) return;
  if (!file.type.startsWith("image/")) {
    showError(`${file.name} is not an image`);
    return;
  }

  busy = true;
  dropzone.classList.add("busy");
  hideError();
  statusLine.textContent = "predicting…";

  try {
    const body = new FormData();
    body.append("image", file, file.name);
    const response = await fetch("/api/predict", { method: "POST", body });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `server returned ${response.status}`);
    }

    const prediction = await response.json();
    await render(file, prediction);
    renderStats(prediction);
    resultBlock.hidden = false;
    statusLine.textContent = `model ready on ${prediction.device}`;
    statusLine.className = "status ready";
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    dropzone.classList.remove("busy");
  }
}

// ---- drawing --------------------------------------------------------------

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("the browser could not display that image"));
    };
    image.src = url;
  });
}

async function render(file, prediction) {
  const image = await loadImage(file);
  const { xmin, ymin, xmax, ymax } = prediction.bbox;

  // Union of the frame and the predicted box, padded.
  const left = Math.min(0, xmin) - PAD;
  const top = Math.min(0, ymin) - PAD;
  const right = Math.max(image.width, xmax) + PAD;
  const bottom = Math.max(image.height, ymax) + PAD;

  const width = Math.round(right - left);
  const height = Math.round(bottom - top);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.aspectRatio = `${width} / ${height}`;

  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.translate(-left, -top);

  context.fillStyle = "#191c20";
  context.fillRect(left, top, width, height);
  context.drawImage(image, 0, 0, image.width, image.height);

  // The frame edge, so it is obvious the box lies outside it.
  context.save();
  context.setLineDash([7, 6]);
  context.strokeStyle = FRAME_STROKE;
  context.lineWidth = 1.5;
  context.globalAlpha = 0.75;
  context.strokeRect(0, 0, image.width, image.height);
  context.restore();

  context.save();
  context.setLineDash([9, 6]);
  context.strokeStyle = PRED_STROKE;
  context.lineWidth = 3;
  context.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
  context.restore();
}

function renderStats(prediction) {
  const box = prediction.bbox;
  const abstained = prediction.direction === -1;

  const rows = [
    ["off-frame side", `${prediction.side_label} (${prediction.side_confidence.toFixed(3)})`],
    [
      "walking direction",
      abstained
        ? "abstained"
        : `${prediction.direction_label} (${prediction.direction_confidence.toFixed(2)})`,
    ],
    ["x range", `${Math.round(box.xmin)} → ${Math.round(box.xmax)}`],
    ["y range", `${Math.round(box.ymin)} → ${Math.round(box.ymax)}`],
    ["box size", `${Math.round(box.xmax - box.xmin)} × ${Math.round(box.ymax - box.ymin)} px`],
    ["image", `${prediction.image_width} × ${prediction.image_height}`],
    ["inference", `${prediction.inference_ms} ms on ${prediction.device}`],
  ];

  statsList.replaceChildren(
    ...rows.map(([label, value]) => {
      const wrapper = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = label;
      const detail = document.createElement("dd");
      detail.textContent = value;
      if (label === "walking direction" && abstained) detail.className = "abstain";
      wrapper.append(term, detail);
      return wrapper;
    })
  );
}

// ---- errors ---------------------------------------------------------------

function showError(message) {
  errorBlock.textContent = message;
  errorBlock.hidden = false;
}
function hideError() {
  errorBlock.hidden = true;
}

checkHealth();
