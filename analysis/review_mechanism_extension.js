/*
 * Read-only mechanism follow-up for the main-batch review addendum.
 *
 * This script reads the frozen seed101 checkpoint, the merged event ledger,
 * the seed101 attribution ledger, and the existing review_checks.json.  It
 * does not fit, train, simulate, calibrate, or alter a protocol.
 *
 * Usage (from the repository root):
 *   node analysis/review_mechanism_extension.js \
 *     --output reports/main_batch_review_addendum/extended_mechanism_checks.json
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const root = path.resolve(arg("--root", path.join(__dirname, "..")));
const output = path.resolve(arg(
  "--output",
  path.join(root, "reports", "main_batch_review_addendum", "extended_mechanism_checks.json")
));
const checkpointPath = path.join(root, "runs", "formal_matrix_main_1000000", "checkpoint_seed101.json");
const eventPath = path.join(root, "data", "main_1000000_full_merged", "events.jsonl");
const attributionPath = path.join(root, "runs", "formal_matrix_main_1000000", "attribution_seed101.jsonl");
const reviewPath = path.join(root, "reports", "main_batch_review_addendum", "review_checks.json");

const EXTREME_UID = "main-1m-20261011:264240";
const TRANSITION_UID = "main-1m-20261011:357493";

async function findRows(file, wanted) {
  const found = {};
  const stream = readline.createInterface({
    input: fs.createReadStream(file, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  for await (const line of stream) {
    if (!line) continue;
    const row = JSON.parse(line);
    if (wanted.has(row.event_uid)) found[row.event_uid] = row;
  }
  return found;
}

function logTailPositive(z) {
  if (!(z > 0)) throw new Error(`positive-tail approximation received z=${z}`);
  const inv = 1 / (z * z);
  // Mills expansion; all follow-up tail evaluations here have z >= 31.
  const correction = 1 - inv + 3 * inv ** 2 - 15 * inv ** 3 + 105 * inv ** 4;
  return -0.5 * z * z - Math.log(z) - 0.5 * Math.log(2 * Math.PI) + Math.log(correction);
}

function logDifference(a, b) {
  if (b > a) throw new Error(`invalid log difference ${a}, ${b}`);
  if (a - b > 40) return a;
  return a + Math.log1p(-Math.exp(b - a));
}

function pionBetheBlochDeltaP(p, thickness, charge) {
  const K = 0.307075;
  const Z_OVER_A = 14.0 / 28.0855;
  const RHO_SI_G_CM3 = 2.329;
  const I_SI_MEV = 0.000173;
  const M_PI_GEV = 0.13957039;
  const mass = M_PI_GEV;
  const energy = Math.sqrt(p * p + mass * mass);
  const beta = p / energy;
  const gamma = energy / mass;
  const me = 0.00051099895;
  const tmax = (2 * me * beta * beta * gamma * gamma) /
    (1 + 2 * gamma * me / mass + (me / mass) ** 2);
  const argument = Math.max(2 * me * beta * beta * gamma * gamma * tmax /
    (I_SI_MEV * I_SI_MEV), 1.0000001);
  const stopping = K * Z_OVER_A * (charge * charge / (beta * beta)) *
    (0.5 * Math.log(argument) - beta * beta);
  const dE = Math.max(stopping * RHO_SI_G_CM3 * (thickness / 10.0) / 1000.0, 0.0);
  const pOut = Math.sqrt(Math.max((energy - dE) ** 2 - mass * mass, 0.0));
  return Math.max(p - pOut, 0.0);
}

function logPdf(z, mu, sigma, upper) {
  // The upper support is many standard deviations above both components for
  // the selected extreme event, so log Phi(upper) is zero to machine precision.
  const u = (upper - mu) / sigma;
  const normalization = u > 8 ? 0 : Math.log(1 - Math.exp(logTailPositive(-u)));
  return -0.5 * ((z - mu) / sigma) ** 2 - Math.log(sigma) -
    0.5 * Math.log(2 * Math.PI) - normalization;
}

function extremeEventCheck(event, attribution, checkpoint) {
  const mean = checkpoint.metadata.target_mean;
  const scale = checkpoint.metadata.target_scale;
  const [logit, offset, mu1, logs0, logs1] = checkpoint.models.joint_anchor_residual_k2.params;
  const pi0 = 1 / (1 + Math.exp(-logit));
  const piR = 1 - pi0;
  const sigma0 = Math.exp(logs0);
  const sigmaR = Math.exp(logs1);
  const z = (event.delta_p_GeV_c - mean) / scale;
  const upper = (event.p_in_GeV_c - mean) / scale;
  const anchorZ = (pionBetheBlochDeltaP(event.p_in_GeV_c, event.thickness_mm, event.pdg / 211.0) - mean) / scale;
  const mu0 = anchorZ + offset;
  const zR = (z - mu1) / sigmaR;
  const upperR = (upper - mu1) / sigmaR;
  const z0 = (z - mu0) / sigma0;
  const upper0 = (upper - mu0) / sigma0;
  const logSurvivalR = logDifference(logTailPositive(zR), logTailPositive(upperR));
  const logSurvival0 = logDifference(logTailPositive(z0), logTailPositive(upper0));
  const logSurvivalM = Math.logaddexp ?
    Math.logaddexp(Math.log(pi0) + logSurvival0, Math.log(piR) + logSurvivalR) :
    Math.log(Math.exp(Math.log(pi0) + logSurvival0) + Math.exp(Math.log(piR) + logSurvivalR));
  const residualLogPdf = logPdf(z, mu1, sigmaR, upper) - Math.log(scale);
  return {
    event_uid: event.event_uid,
    p_in_GeV_c: event.p_in_GeV_c,
    delta_p_GeV_c: event.delta_p_GeV_c,
    standardized_target_z: z,
    standardized_upper_support: upper,
    anchor_loss_GeV_c: pionBetheBlochDeltaP(event.p_in_GeV_c, event.thickness_mm, event.pdg / 211.0),
    joint_anchor_mu_z: mu0,
    residual_mu_z: mu1,
    residual_sigma_z: sigmaR,
    residual_standardized_distance: zR,
    residual_nll_original_units: -residualLogPdf,
    log_survival_anchor: logSurvival0,
    log_survival_residual: logSurvivalR,
    log_survival_mixture: logSurvivalM,
    survival_mixture: Math.exp(logSurvivalM),
    saved_log_f_full: attribution.log_f_full,
    saved_nll_mixture: -attribution.log_f_full,
    saved_log_responsibility_anchor: attribution.log_responsibility0,
  };
}

function getPath(object, parts) {
  return parts.reduce((value, part) => value[part], object);
}

async function main() {
  const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, "utf8"));
  const review = JSON.parse(fs.readFileSync(reviewPath, "utf8"));
  const wanted = new Set([EXTREME_UID, TRANSITION_UID]);
  const events = await findRows(eventPath, wanted);
  const attributions = await findRows(attributionPath, wanted);
  const extreme = extremeEventCheck(events[EXTREME_UID], attributions[EXTREME_UID], checkpoint);
  const transition = attributions[TRANSITION_UID];
  const logPi0 = transition.log_pi0;
  const pi0 = Math.exp(logPi0);
  const piR = 1 - pi0;
  const gateOnlyBound = -Math.log(piR);
  const transitionUpper = transition.G_total + gateOnlyBound;
  const component = getPath(review, ["region_attribution_seed101", "groups", "diagnostic", "all", "component_position_scale_summary"]);
  const mu0Min = component.min_max_with_event_uid.mu0_z[0];
  const mu0Max = component.min_max_with_event_uid.mu0_z[2];
  const mu0Mid = (mu0Min + mu0Max) / 2;
  const sigmaA = checkpoint.models.joint_anchor_residual_k2.params[3] ?
    Math.exp(checkpoint.models.joint_anchor_residual_k2.params[3]) : null;
  const sigmaR = Math.exp(checkpoint.models.joint_anchor_residual_k2.params[4]);
  const withinA = pi0 * sigmaA * sigmaA;
  const withinR = piR * sigmaR * sigmaR;
  const separation = pi0 * piR * (checkpoint.models.joint_anchor_residual_k2.params[2] - mu0Mid) ** 2;
  const totalVariance = withinA + withinR + separation;
  const q99MaxLogResp = getPath(review, ["region_attribution_seed101", "groups", "diagnostic", "q99_q999", "min_max_with_event_uid", "log_responsibility0"])[2];
  const gradientFactor = Math.exp(q99MaxLogResp);

  const result = {
    schema: "pion-material-study.mechanism-followup.v1",
    read_only: true,
    source_files: {
      checkpoint: path.relative(root, checkpointPath).replaceAll("\\", "/"),
      merged_events: path.relative(root, eventPath).replaceAll("\\", "/"),
      attribution: path.relative(root, attributionPath).replaceAll("\\", "/"),
      prior_review_checks: path.relative(root, reviewPath).replaceAll("\\", "/"),
    },
    extreme_event: extreme,
    global_gate_and_scales: {
      log_pi0: logPi0,
      pi0,
      pi_residual: piR,
      sigma_anchor_z: sigmaA,
      sigma_residual_z: sigmaR,
      sigma_ratio_residual_over_anchor: sigmaR / sigmaA,
      q99_q999_max_log_responsibility_anchor: q99MaxLogResp,
      q99_q999_min_residual_responsibility: 1 - gradientFactor,
      anchor_gradient_factor_upper_bound: gradientFactor,
      anchor_gradient_attenuation_lower_bound: 1 / gradientFactor,
    },
    gate_only_transition_bound: {
      event_uid: TRANSITION_UID,
      saved_G_total: transition.G_total,
      maximum_log_score_improvement_over_fixed_components: gateOnlyBound,
      upper_bound_G_after_arbitrary_global_weight_change: transitionUpper,
      upper_bound_density_ratio_to_reference: Math.exp(transitionUpper),
      interpretation: "A global weight-only reallocation cannot make this fixed-component transition event positive relative to the reference.",
    },
    approximate_standardized_variance_budget: {
      anchor_mu_z_range: [mu0Min, mu0Max],
      anchor_mu_z_used_for_between_component_term: mu0Mid,
      within_anchor_term: withinA,
      within_residual_term: withinR,
      between_component_center_term: separation,
      total_mixture_variance_approximation: totalVariance,
      residual_within_fraction_of_within_terms: withinR / (withinA + withinR),
      residual_plus_center_fraction_of_total: (withinR + separation) / totalVariance,
      caveat: "Moment budget uses the fitted Gaussian mother parameters in standardized coordinates and ignores truncation corrections; it is a scale diagnostic, not a generated-data variance estimate.",
    },
    interpretation_limits: [
      "Posterior responsibility is model-internal explanation, not an input-dependent gate switch.",
      "The gradient attenuation calculation is a likelihood sensitivity statement, not proof of causal retraining dynamics.",
      "These checks do not identify a unique microscopic process and do not establish calibration.",
    ],
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({ output, extreme_event: extreme.event_uid, model_survival: extreme.survival_mixture }));
}

main().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
