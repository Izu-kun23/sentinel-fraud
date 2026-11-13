const pct = (n) => (n * 100).toFixed(1) + "%";
async function load() {
  message(
    "Training three models on synthetic data. First load may take a moment…",
  );
  const d = await api("/overview");
  $("data-notice").textContent =
    d.source +
    " · Scores are model estimates; independent validation is required before real financial decisions.";
  const m = d.metrics[d.winner];
  stats([
    ["Test transactions", d.transactions, "20% untouched holdout"],
    ["Flagged for review", d.flagged, "Validation-selected threshold"],
    [
      "Average precision",
      pct(m.average_precision),
      "Selected model, held-out test",
    ],
    ["Fraud prevalence", pct(d.prevalence), "Synthetic population"],
  ]);
  $("models").innerHTML = Object.entries(d.metrics)
    .map(
      ([n, m]) =>
        `<tr><td><strong>${esc(n)}</strong></td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${m.f1.toFixed(3)}</td><td>${m.average_precision.toFixed(3)}</td></tr>`,
    )
    .join("");
  $("winner").textContent =
    "Selected by validation average precision: " +
    d.winner +
    ". Test false-positive rate: " +
    pct(m.false_positive_rate);
  $("transactions").innerHTML = d.rows
    .map(
      (r) =>
        `<tr><td>${r.id}</td><td>£${r.amount.toFixed(2)}</td><td>${r.distance_km.toFixed(0)} km</td><td>${pct(r.probability)}</td><td><span class="pill">${r.flagged ? "Review" : "Below threshold"}</span></td></tr>`,
    )
    .join("");
  $("histogram").innerHTML = d.histogram
    .map(
      (n, i) =>
        `<div class="bar" style="height:${Math.max(3, (n / Math.max(...d.histogram)) * 100)}%" title="Bin ${i + 1}: ${n} transactions"></div>`,
    )
    .join("");
  message("Models ready. Inspect a transaction to see SHAP explanations.");
}
$("predict").onsubmit = (e) => {
  e.preventDefault();
  busy(e.submitter, async () => {
    const data = Object.fromEntries(
      [
        "amount",
        "hour",
        "distance_km",
        "transactions_24h",
        "account_age_days",
        "international",
      ].map((k) => [k, Number($(k).value)]),
    );
    const d = await api("/predict", {
      method: "POST",
      body: JSON.stringify(data),
    });
    $("prediction").innerHTML =
      `<div class="divider"></div><span class="pill">${d.flagged ? "FLAGGED FOR REVIEW" : "BELOW REVIEW THRESHOLD"}</span><div class="score">${pct(d.probability)}</div><p class="muted">${esc(d.model)} · threshold ${pct(d.threshold)}</p><h3>What influenced this score?</h3><p class="muted">SHAP contributions in ${esc(d.explanation.output_space)}. Positive values raise the score.</p>` +
      d.explanation.contributions
        .map(
          (f) =>
            `<div class="feature"><span>${esc(f.feature.replaceAll("_", " "))}</span><strong>${f.value > 0 ? "+" : ""}${f.value.toFixed(3)}</strong></div>`,
        )
        .join("");
    message(
      "Analysis complete. Explanations describe model behaviour, not causation.",
    );
  });
};
load().catch((e) => message(e.message, true));

$("import").onsubmit = (e) => {
  e.preventDefault();
  busy(e.submitter, async () => {
    const file = $("csv").files[0];
    if (!file || file.size > 1000000)
      throw Error("Select a CSV smaller than 1 MB");
    const d = await api("/transactions/import", {
      method: "POST",
      body: JSON.stringify({ csv: await file.text() }),
    });
    $("transactions").innerHTML = d.rows
      .map(
        (r) =>
          `<tr><td>${r.id}</td><td>£${r.amount.toFixed(2)}</td><td>${r.distance_km.toFixed(0)} km</td><td>${pct(r.probability)}</td><td><span class="pill">${r.flagged ? "Review" : "Below threshold"}</span></td></tr>`,
      )
      .join("");
    message(
      d.count +
        " imported transactions scored; " +
        d.flagged +
        " flagged. " +
        d.note,
    );
  });
};
