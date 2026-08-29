// ac_plant.mjs — the two training-plant behaviours that cannot live in the
// XML because they are state-dependent, implemented exactly where mjlab/BAM
// implement them: around every physics substep.
//
// 1. ACTUATION DELAY. FrictionDRBamActuatorCfg(delay_min_lag=3,
//    delay_max_lag=6) — units are PHYSICS timesteps (mjlab actuator.py
//    docstring), the lag is resampled uniformly per physics step
//    (delay_update_period=0, hold_prob=0), and it is nominal for EVERY task,
//    not a DR event. So the position target MuJoCo sees is the one the policy
//    issued 15–30 ms ago. N = Uniform{3,4,5,6} physics steps, resampled each
//    substep from a seeded xorshift so recordings are reproducible.
//
// 2. BAM m6 FRICTION BUDGET. Training zeroes the XML damping/frictionloss and
//    rewrites dof_frictionloss every substep with Coulomb + Stribeck +
//    directional load friction + quadratic terms (bam/mjlab.py
//    _compute_friction_budget, params xl330/m6.json). Here the same budget is
//    written into model.dof_frictionloss before each mj_step. Two deliberate
//    deviations from training, both stated:
//      - dof_damping stays at the XML's 0.053 (= back-EMF kt²/R 0.0477 +
//        friction_viscous 0.0054) because the position-servo surrogate has no
//        motor torque law to carry the back-EMF; training has damping 0.00536
//        with back-EMF inside the motor torque. Same linearization BAM's own
//        to_mujoco() publishes.
//      - motor-side load = data.qfrc_actuator of the previous solve (exactly
//        what BAM uses); external load = -qfrc_bias + qfrc_constraint WITHOUT
//        stripping the dof-friction constraint force (BAM strips it via an
//        efc scan the WASM API does not expose). The error is bounded by the
//        frictionloss value itself and only enters through the small
//        external-side coefficients (8.5e-6, and 0.081 gated by Stribeck).
//
// xl330 m6 parameters, verbatim from Rhoban/bam@62bd8ce bam/params/xl330/m6.json.
const P = {
  friction_base: 0.004771183165566,
  friction_stribeck: 0.004676345799486616,
  lf_motor: 0.2667860954283698,
  lf_ext: 8.515871897059342e-06,
  lf_motor_str: 1.0722918395099123e-05,
  lf_ext_str: 0.08077928978935671,
  lf_motor_quad: 0.009972471242139415,
  lf_ext_quad: 0.004902565732332559,
  dtheta: 2.890372094130307,
  alpha: 8.683259907618984,
};

export const DELAY_MIN = 3, DELAY_MAX = 6; // physics timesteps (15–30 ms)

export function makePlant(mj, model, data, D, seed = 0x51ed270b, opts = {}) {
  const useDelay = opts.delay ?? true;       // ablation switches; default = full plant
  const useFriction = opts.friction ?? true;
  let s = seed >>> 0;
  const rand = () => {
    s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
    return s / 0x100000000;
  };

  // Ring of per-substep ctrl targets, newest last. Depth covers DELAY_MAX.
  let hist = [];
  function resetDelay(target) {
    hist = [];
    for (let i = 0; i <= DELAY_MAX; i++) hist.push(target.slice());
  }

  function updateFriction() {
    for (let k = 0; k < 14; k++) {
      const d = D.dof[k];
      const av = Math.abs(data.qvel[d]);
      const st = Math.exp(-Math.pow(av / P.dtheta, P.alpha));
      const mot = data.qfrc_actuator[d];               // previous solve, as BAM
      const ext = -data.qfrc_bias[d] + data.qfrc_constraint[d];
      const absM = Math.abs(mot), absE = Math.abs(ext);
      model.dof_frictionloss[d] =
        P.friction_base
        + st * P.friction_stribeck
        + Math.abs(ext * P.lf_ext - mot * P.lf_motor)
        + st * Math.abs(ext * P.lf_ext_str - mot * P.lf_motor_str)
        + st * (absM > absE ? P.lf_ext_quad * absE * absE
                            : P.lf_motor_quad * absM * absM);
    }
  }

  /**
   * One 50 Hz policy tick: 4 physics substeps of 0.005 s, each seeing the
   * DELAYED position target and the fresh friction budget.
   */
  function stepTick(target) {
    for (let sub = 0; sub < 4; sub++) {
      hist.push(target.slice());
      if (hist.length > DELAY_MAX + 1) hist.shift();
      const lag = useDelay
        ? DELAY_MIN + Math.floor(rand() * (DELAY_MAX - DELAY_MIN + 1))
        : 0;
      const src = hist[Math.max(0, hist.length - 1 - lag)];
      for (let k = 0; k < 14; k++) data.ctrl[k] = src[k];
      if (useFriction) updateFriction();
      mj.mj_step(model, data);
    }
  }

  return { stepTick, resetDelay, rand };
}
