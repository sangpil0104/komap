const samples = [
  "A117_step4_sample2",
  "A11_step1_sample4",
  "A11_step2_sample4",
  "A125_step1_sample4",
  "A153_step3_sample3",
  "A170_step4_sample3",
  "A181_step1_sample1",
  "A186_step2_sample2",
  "A191_step4_sample3",
  "A40_step3_sample3",
  "A42_step2_sample4",
  "A44_step4_sample3",
  "A55_step2_sample1",
  "A59_step3_sample1",
  "A63_step1_sample4",
  "A67_step1_sample1",
  "A71_step2_sample1",
  "A72_step4_sample4",
  "A74_step3_sample4",
  "A7_step1_sample4",
];

const state = { index: 0, records: new Map() };
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

function paths(name) {
  return {
    original: `assets/valid_original/${name}_image.png`,
    groundTruth: `assets/valid_ground_truth/${name}_mask.png`,
    prediction: `assets/valid_prediction/${name}_mask.png`,
    errorMap: `assets/valid_error/${name}_error.png`,
  };
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderMetrics(name) {
  const record = state.records.get(name);
  document.querySelector("#selected-name").textContent = name;
  document.querySelector("#selected-miou").textContent = record ? percent(record.miou) : "loading";
  document.querySelector("#selected-error-rate").textContent = record ? percent(record.error_rate) : "loading";
  document.querySelector("#selected-error-pixels").textContent = record ? record.error_pixels.toLocaleString() : "loading";
  document.querySelector("#selected-pixels").textContent = record ? record.pixels.toLocaleString() : "loading";
  document.querySelector("#error-summary").textContent = record ? `wrong ${percent(record.error_rate)}` : "loading";
}

function render(index) {
  state.index = (index + samples.length) % samples.length;
  const name = samples[state.index];
  const path = paths(name);
  original.src = path.original;
  groundTruth.src = path.groundTruth;
  prediction.src = path.prediction;
  errorMap.src = path.errorMap;
  overlayOriginal.src = path.original;
  overlayPrediction.src = path.prediction;
  select.value = name;
  counter.textContent = `${state.index + 1} / ${samples.length}`;
  document.querySelectorAll(".thumb").forEach((item, itemIndex) => item.classList.toggle("active", itemIndex === state.index));
  renderMetrics(name);
  original.onload = () => {
    size.textContent = `${original.naturalWidth} × ${original.naturalHeight}`;
    document.querySelector("#overlay-frame").style.aspectRatio = `${original.naturalWidth} / ${original.naturalHeight}`;
  };
}

function buildControls() {
  samples.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${index + 1}. ${name}`;
    select.appendChild(option);

    const button = document.createElement("button");
    button.className = "thumb";
    button.type = "button";
    button.innerHTML = `<img src="${paths(name).original}" alt=""><span>${index + 1}. ${name}</span>`;
    button.addEventListener("click", () => render(index));
    thumbs.appendChild(button);
  });
}

function renderGlobal(manifest) {
  const global = manifest.global;
  document.querySelector("#global-miou").textContent = global ? global.global_miou.toFixed(4) : "—";
  document.querySelector("#image-mean-miou").textContent = global ? global.image_mean_miou.toFixed(4) : "—";
  if (!global || !global.global_iou) return;
  document.querySelector("#global-primary").textContent = global.global_iou["Primary Si"].toFixed(4);
  document.querySelector("#global-al3ni").textContent = global.global_iou.Al3Ni.toFixed(4);
  document.querySelector("#global-eutectic").textContent = global.global_iou["Eutectic Si"].toFixed(4);
  document.querySelector("#global-al").textContent = global.global_iou.Al.toFixed(4);
}

document.querySelector("#prev").addEventListener("click", () => render(state.index - 1));
document.querySelector("#next").addEventListener("click", () => render(state.index + 1));
select.addEventListener("change", (event) => render(samples.indexOf(event.target.value)));
opacity.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  overlayPrediction.style.opacity = value / 100;
  opacityValue.textContent = `${value}%`;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") render(state.index - 1);
  if (event.key === "ArrowRight") render(state.index + 1);
});

buildControls();
render(0);
fetch("assets/valid_manifest.json")
  .then((response) => response.json())
  .then((manifest) => {
    (manifest.records || []).forEach((record) => state.records.set(record.name, record));
    renderGlobal(manifest);
    render(state.index);
  })
  .catch(() => {
    document.querySelector("#selected-name").textContent = "manifest를 불러오지 못했습니다";
  });
