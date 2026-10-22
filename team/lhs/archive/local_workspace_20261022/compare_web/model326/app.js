const state = { index: 0, records: [], byName: new Map() };
const original = document.querySelector("#original");
const groundTruth = document.querySelector("#ground-truth");
const prediction = document.querySelector("#prediction");
const errorMap = document.querySelector("#error-map");
const overlayOriginal = document.querySelector("#overlay-original");
const overlayPrediction = document.querySelector("#overlay-prediction");
const select = document.querySelector("#image-select");
const counter = document.querySelector("#counter");
const size = document.querySelector("#original-size");
const opacity = document.querySelector("#opacity");
const opacityValue = document.querySelector("#opacity-value");
const thumbs = document.querySelector("#thumbnails");

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function paths(record) {
  return {
    original: `../assets/valid_original/${record.image}`,
    groundTruth: `../assets/valid_ground_truth/${record.name}_mask.png`,
    prediction: `assets/valid_prediction/${record.name}_mask.png`,
    errorMap: `assets/valid_error/${record.name}_error.png`,
  };
}

function renderMetrics(record) {
  document.querySelector("#selected-name").textContent = record.name;
  document.querySelector("#selected-miou").textContent = percent(record.miou);
  document.querySelector("#selected-error-rate").textContent = percent(record.error_rate);
  document.querySelector("#selected-error-pixels").textContent = record.error_pixels.toLocaleString();
  document.querySelector("#selected-pixels").textContent = record.pixels.toLocaleString();
  document.querySelector("#error-summary").textContent = `오류 ${percent(record.error_rate)}`;
  document.querySelector("#iou-al3ni").textContent = percent(record.iou_by_class["Al3Ni"]);
  document.querySelector("#iou-primary").textContent = percent(record.iou_by_class["Primary Si"]);
  document.querySelector("#iou-al").textContent = percent(record.iou_by_class.Al);
  document.querySelector("#iou-eutectic").textContent = percent(record.iou_by_class["Eutectic Si"]);
}

function render(index) {
  state.index = (index + state.records.length) % state.records.length;
  const record = state.records[state.index];
  const path = paths(record);
  original.src = path.original;
  groundTruth.src = path.groundTruth;
  prediction.src = path.prediction;
  errorMap.src = path.errorMap;
  overlayOriginal.src = path.original;
  overlayPrediction.src = path.prediction;
  select.value = record.name;
  counter.textContent = `${state.index + 1} / ${state.records.length}`;
  document.querySelectorAll(".thumb").forEach((item, itemIndex) =>
    item.classList.toggle("active", itemIndex === state.index));
  renderMetrics(record);
  original.onload = () => {
    size.textContent = `${original.naturalWidth} × ${original.naturalHeight}`;
    document.querySelector("#overlay-frame").style.aspectRatio = `${original.naturalWidth} / ${original.naturalHeight}`;
  };
}

function buildControls() {
  state.records.forEach((record, index) => {
    const option = document.createElement("option");
    option.value = record.name;
    option.textContent = `${index + 1}. ${record.name} · ${percent(record.miou)}`;
    select.appendChild(option);

    const button = document.createElement("button");
    button.className = "thumb";
    button.type = "button";
    button.innerHTML = `<img src="${paths(record).original}" alt=""><span>${index + 1}. ${record.name} · ${percent(record.miou)}</span>`;
    button.addEventListener("click", () => render(index));
    thumbs.appendChild(button);
  });
}

function renderGlobal(global) {
  document.querySelector("#global-miou").textContent = `${(global.global_miou * 100).toFixed(2)}%`;
  document.querySelector("#image-mean-miou").textContent = `${(global.image_mean_miou * 100).toFixed(2)}%`;
  document.querySelector("#global-al3ni").textContent = percent(global.global_iou["Al3Ni"]);
  document.querySelector("#global-primary").textContent = percent(global.global_iou["Primary Si"]);
  document.querySelector("#global-al").textContent = percent(global.global_iou.Al);
  document.querySelector("#global-eutectic").textContent = percent(global.global_iou["Eutectic Si"]);
}

document.querySelector("#prev").addEventListener("click", () => render(state.index - 1));
document.querySelector("#next").addEventListener("click", () => render(state.index + 1));
select.addEventListener("change", (event) => {
  render(state.records.findIndex((record) => record.name === event.target.value));
});
opacity.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  overlayPrediction.style.opacity = value / 100;
  opacityValue.textContent = `${value}%`;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") render(state.index - 1);
  if (event.key === "ArrowRight") render(state.index + 1);
});

fetch("assets/valid_manifest.json")
  .then((response) => {
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    return response.json();
  })
  .then((manifest) => {
    state.records = manifest.records;
    state.byName = new Map(state.records.map((record) => [record.name, record]));
    buildControls();
    renderGlobal(manifest.global);
    render(0);
  })
  .catch((error) => {
    document.querySelector("#selected-name").textContent = `manifest 오류: ${error.message}`;
  });
