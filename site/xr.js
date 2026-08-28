// WebXR: put the duck on your actual floor.
//
// The renderer is already plain WebGL with an explicit view and projection
// matrix per frame, so an immersive session is mostly a matter of letting the
// headset supply those matrices instead of the chase camera. That is the whole
// reason this was not written with three.js.
//
// AR first, VR as a fallback. `immersive-ar` with hit-testing puts the duck on
// a real surface; `immersive-vr` drops it on a virtual floor for headsets with
// no passthrough. Safari supports neither today, which is why the iOS path in
// this project is a native one.
export async function xrSupport() {
  if (!navigator.xr) return { ar: false, vr: false, reason: 'no WebXR in this browser' };
  const [ar, vr] = await Promise.all([
    navigator.xr.isSessionSupported('immersive-ar').catch(() => false),
    navigator.xr.isSessionSupported('immersive-vr').catch(() => false),
  ]);
  return { ar, vr, reason: ar || vr ? null : 'no immersive session available' };
}

/**
 * Start a session and drive it.
 *
 * `onFrame({ view, proj, origin })` is called once per eye per frame with the
 * headset's own matrices. `origin` is where the duck should stand — the hit-test
 * result in AR, or a fixed spot ahead of you in VR.
 */
export async function startXR({ gl, mode, onFrame, onEnd, step }) {
  const session = await navigator.xr.requestSession(mode, {
    requiredFeatures: mode === 'immersive-ar' ? ['local-floor'] : ['local-floor'],
    optionalFeatures: mode === 'immersive-ar' ? ['hit-test'] : [],
  });
  await gl.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const refSpace = await session.requestReferenceSpace('local-floor');

  // Hit-testing puts the duck where you are looking, once, on the first frame
  // that finds a surface. After that it stays put — a duck that re-anchored
  // every frame would slide around the room.
  let hitSource = null, placed = null;
  if (mode === 'immersive-ar' && session.requestHitTestSource) {
    try {
      const viewer = await session.requestReferenceSpace('viewer');
      hitSource = await session.requestHitTestSource({ space: viewer });
    } catch { /* hit-test is optional; fall back to a fixed spot */ }
  }

  session.addEventListener('end', () => onEnd?.());

  function frame(t, xrFrame) {
    session.requestAnimationFrame(frame);
    const pose = xrFrame.getViewerPose(refSpace);
    if (!pose) return;

    if (!placed) {
      if (hitSource) {
        const hits = xrFrame.getHitTestResults(hitSource);
        if (hits.length) {
          const p = hits[0].getPose(refSpace);
          placed = [p.transform.position.x, p.transform.position.y, p.transform.position.z];
        }
      } else {
        placed = [0, 0, -0.8];      // a metre or so in front, on the floor
      }
    }

    step?.(t);

    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    for (const view of pose.views) {
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      onFrame({
        view: view.transform.inverse.matrix,
        proj: view.projectionMatrix,
        origin: placed || [0, 0, -0.8],
      });
    }
  }
  session.requestAnimationFrame(frame);
  return session;
}
