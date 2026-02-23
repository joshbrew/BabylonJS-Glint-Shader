// glint/centeredOctaveGaussianGlint.js
//
// Reusable Babylon.js ShaderMaterial module for the centered octave gaussian glint shader.
// This version is ESM and includes:
// - shader source installation
// - a structured configuration model
// - config validation/clamping
// - legacy flat-params compatibility helpers
// - material creation
// - per-frame uniform upload
// - render scale application helper
//
// The shader itself is GLSL and intended for Babylon's WebGPU path via Babylon's shader translation stack.

import {
  Effect,
  ShaderMaterial,
  Color3,
  Vector3,
  ShaderLanguage,
} from "@babylonjs/core";

/**
 * Stable shader key used in Babylon's Effect.ShadersStore.
 * You can override this name when installing or creating the material.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_SHADER_NAME =
  "centeredOctaveGaussianGlintShardAnisoHybridStable";

/**
 * Uniform list declared for the ShaderMaterial.
 * Babylon needs this to bind explicit uniforms used by the GLSL shaders.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_UNIFORMS = [
  "world",
  "worldViewProjection",
  "uCameraPos",
  "uLightDir",

  "uDensity01",
  "uRoughness01",
  "uMicrofacetRoughness",
  "uPixelFilterSize",
  "uUVScale",
  "uIntensity",

  "uSigmaScale",
  "uSigmaMinCell",
  "uSigmaMaxCell",

  "uAnisoWarpStrength",

  "uShardMix",
  "uShardSharpness",
  "uShardEdgeHardness",
  "uShardClipStrength",

  "uUseToneMapGamma",
  "uUseSparkleColorGradient",
  "uUseAnisoWarpMode",
  "uUseGlassShardMode",

  "uFrontLightColor",
  "uFrontLightIntensity",
  "uBackLightColor",
  "uBackLightIntensity",
];

/**
 * Optional schema metadata for building UI controls or documentation.
 * This is descriptive and not enforced automatically by Babylon.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_CONFIG_SCHEMA = {
  glint: {
    density01: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.001,
      uniform: "uDensity01",
      label: "density (active cell keep)",
      description: "Keep probability for active cells. Does not change cell size.",
    },
    roughness01: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.001,
      uniform: "uRoughness01",
      label: "roughness",
      description: "Macro roughness control for GGX alpha remap.",
    },
    microfacetRoughness: {
      type: "float",
      min: 0.001,
      max: 0.1,
      step: 0.001,
      uniform: "uMicrofacetRoughness",
      label: "microfacet roughness",
      description: "Angular gaussian spread in the glint NDF domain.",
    },
    pixelFilterSize: {
      type: "float",
      min: 0.4,
      max: 1.4,
      step: 0.01,
      uniform: "uPixelFilterSize",
      label: "pixel filter size",
      description: "Derivative footprint scale for anti-aliasing of cell gaussians.",
    },
    uvScale: {
      type: "float",
      min: 0.01,
      max: 12.0,
      step: 0.01,
      uniform: "uUVScale",
      label: "domain scale",
      description: "Pattern scale in object-local projected domain.",
    },
    intensity: {
      type: "float",
      min: 0.0,
      max: 5.0,
      step: 0.01,
      uniform: "uIntensity",
      label: "glint intensity",
      description: "Shader multiplies this by an internal 10x gain.",
    },
  },

  sigma: {
    scale: {
      type: "float",
      min: 0.25,
      max: 2.5,
      step: 0.01,
      uniform: "uSigmaScale",
      label: "gaussian sigma scale",
      description: "Global scale applied to gaussian sigma in cell units.",
    },
    minCell: {
      type: "float",
      min: 0.01,
      max: 0.2,
      step: 0.001,
      uniform: "uSigmaMinCell",
      label: "sigma min clamp (cell units)",
      description: "Minimum sigma clamp in local cell units.",
    },
    maxCell: {
      type: "float",
      min: 0.05,
      max: 0.49,
      step: 0.001,
      uniform: "uSigmaMaxCell",
      label: "sigma max clamp (cell units)",
      description: "Maximum sigma clamp in local cell units.",
    },
  },

  anisoWarp: {
    enabled: {
      type: "bool",
      uniform: "uUseAnisoWarpMode",
      label: "aniso warp mode",
      description: "Enables in-cell anisotropic lens-like warp.",
    },
    strength: {
      type: "float",
      min: 0.0,
      max: 1.5,
      step: 0.01,
      uniform: "uAnisoWarpStrength",
      label: "aniso warp strength",
      description: "Strength used when aniso warp mode is enabled.",
    },
  },

  glassShard: {
    enabled: {
      type: "bool",
      uniform: "uUseGlassShardMode",
      label: "glass shard mode",
      description: "Enables hard clipped shard-like gaussian silhouette.",
    },
    mix: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.01,
      uniform: "uShardMix",
      label: "glass shard mix",
      description: "Shard silhouette blend amount.",
    },
    sharpness: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.01,
      uniform: "uShardSharpness",
      label: "shard sharpness",
      description: "Controls shard aspect and superellipse shaping.",
    },
    edgeHardness: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.01,
      uniform: "uShardEdgeHardness",
      label: "shard edge hardness",
      description: "Controls clip edge transition softness.",
    },
    clipStrength: {
      type: "float",
      min: 0.0,
      max: 1.0,
      step: 0.01,
      uniform: "uShardClipStrength",
      label: "shard clip strength",
      description: "Strength of shard clipping on gaussian contribution.",
    },
  },

  color: {
    sparkleGradient: {
      type: "bool",
      uniform: "uUseSparkleColorGradient",
      label: "random sparkle colors",
      description: "Analytic palette color per sparkle cell.",
    },
  },

  post: {
    toneMapGamma: {
      type: "bool",
      uniform: "uUseToneMapGamma",
      label: "tonemap + gamma",
      description: "Applies simple Reinhard tonemap and gamma.",
    },
  },

  lighting: {
    direction: {
      type: "vec3",
      uniform: "uLightDir",
      label: "light direction",
      description: "World-space light direction, normalized before upload.",
    },
    frontColor: {
      type: "vec3",
      uniform: "uFrontLightColor",
      label: "front light color",
      description: "Front-facing light tint.",
    },
    frontIntensity: {
      type: "float",
      uniform: "uFrontLightIntensity",
      label: "front light intensity",
      description: "Front-facing light intensity multiplier.",
    },
    backColor: {
      type: "vec3",
      uniform: "uBackLightColor",
      label: "back light color",
      description: "Back light tint for opposite-side highlights.",
    },
    backIntensity: {
      type: "float",
      uniform: "uBackLightIntensity",
      label: "back light intensity",
      description: "Back light intensity multiplier.",
    },
  },
};

/**
 * Nested default config used by the reusable API.
 * This is the preferred config shape for new code.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_DEFAULT_CONFIG = Object.freeze({
  shader: {
    name: CENTERED_OCTAVE_GAUSSIAN_GLINT_SHADER_NAME,
    language: "GLSL",
  },

  material: {
    backFaceCulling: true,
  },

  glint: {
    density01: 0.42,
    roughness01: 0.28,
    microfacetRoughness: 0.014,
    pixelFilterSize: 0.62,
    uvScale: 1.65,
    intensity: 1.1,
  },

  sigma: {
    scale: 1.0,
    minCell: 0.045,
    maxCell: 0.22,
  },

  anisoWarp: {
    enabled: false,
    strength: 0.6,
  },

  glassShard: {
    enabled: false,
    mix: 0.75,
    sharpness: 0.88,
    edgeHardness: 0.92,
    clipStrength: 0.95,
  },

  color: {
    sparkleGradient: true,
  },

  post: {
    toneMapGamma: false,
  },

  lighting: {
    direction: [1, 1, 1],
    front: {
      color: [0.95, 0.68, 0.48],
      intensity: 8.0,
    },
    back: {
      color: [0.12, 0.16, 0.24],
      intensity: 4.0,
    },
  },

  render: {
    renderScale: 0.8,
  },

  motion: {
    spinMeshes: true,
    autoOrbit: true,
    orbitSpeed: 0.2,
  },
});

/**
 * Flat legacy defaults matching the shape used in the original demo page.
 * Useful when migrating existing UI code incrementally.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_LEGACY_DEFAULTS = Object.freeze({
  density01: 0.42,
  roughness01: 0.28,
  microfacetRoughness: 0.014,
  pixelFilterSize: 0.62,
  uvScale: 1.65,
  intensity: 1.1,

  sigmaScale: 1.0,
  sigmaMinCell: 0.045,
  sigmaMaxCell: 0.22,

  anisoWarpStrength: 0.6,

  shardMix: 0.75,
  shardSharpness: 0.88,
  shardEdgeHardness: 0.92,
  shardClipStrength: 0.95,

  renderScale: 0.8,
  orbitSpeed: 0.2,

  spinMeshes: true,
  autoOrbit: true,
  toneMapGamma: false,
  sparkleColorGradient: true,
  anisoWarpMode: false,
  glassShardMode: false,
});

/**
 * Deep clone helper for plain data trees.
 * This is sufficient for our config objects (numbers, booleans, arrays, objects).
 */
function deepClone(x) {
  if (Array.isArray(x)) return x.map(deepClone);
  if (x && typeof x === "object") {
    const out = {};
    for (const k of Object.keys(x)) out[k] = deepClone(x[k]);
    return out;
  }
  return x;
}

/**
 * Plain object check used by the deep merge helper.
 */
function isPlainObject(x) {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

/**
 * Deep merge helper.
 * - base is cloned
 * - override values replace base values
 * - nested objects merge recursively
 */
function deepMerge(base, override) {
  if (!isPlainObject(base)) return deepClone(override);
  const out = deepClone(base);
  if (!isPlainObject(override)) return out;

  for (const k of Object.keys(override)) {
    const bv = out[k];
    const ov = override[k];
    if (isPlainObject(bv) && isPlainObject(ov)) out[k] = deepMerge(bv, ov);
    else out[k] = deepClone(ov);
  }
  return out;
}

/**
 * Numeric clamp helper.
 */
function clamp(x, a, b) {
  return Math.min(b, Math.max(a, x));
}

/**
 * Normalize a 3-component array.
 * Falls back to [0, 0, 1] if invalid or zero-length.
 */
function normalize3(v) {
  const x = Number(v?.[0] ?? 0);
  const y = Number(v?.[1] ?? 0);
  const z = Number(v?.[2] ?? 1);
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

/**
 * Convert an array [r,g,b] to Babylon Color3.
 */
function toColor3(v, fallback) {
  const a = Array.isArray(v) && v.length >= 3 ? v : fallback;
  return new Color3(Number(a[0]), Number(a[1]), Number(a[2]));
}

/**
 * Convert an array [x,y,z] to Babylon Vector3.
 * Optionally normalizes the vector before returning.
 */
function toVector3(v, fallback, normalize = false) {
  const a = Array.isArray(v) && v.length >= 3 ? v : fallback;
  const vec = new Vector3(Number(a[0]), Number(a[1]), Number(a[2]));
  return normalize ? vec.normalize() : vec;
}

/**
 * Create a validated, clamped config object.
 * This is the main entry point for config sanitization.
 */
export function createCenteredOctaveGaussianGlintConfig(overrides = {}) {
  const cfg = deepMerge(CENTERED_OCTAVE_GAUSSIAN_GLINT_DEFAULT_CONFIG, overrides);

  // Glint controls
  cfg.glint.density01 = clamp(Number(cfg.glint.density01), 0, 1);
  cfg.glint.roughness01 = clamp(Number(cfg.glint.roughness01), 0, 1);
  cfg.glint.microfacetRoughness = clamp(
    Number(cfg.glint.microfacetRoughness),
    0.001,
    0.1,
  );
  cfg.glint.pixelFilterSize = clamp(Number(cfg.glint.pixelFilterSize), 0.1, 4.0);
  cfg.glint.uvScale = Math.max(0.0001, Number(cfg.glint.uvScale));
  cfg.glint.intensity = Math.max(0, Number(cfg.glint.intensity));

  // Sigma controls
  cfg.sigma.scale = Math.max(0.01, Number(cfg.sigma.scale));
  cfg.sigma.minCell = clamp(Number(cfg.sigma.minCell), 0.005, 0.49);
  cfg.sigma.maxCell = clamp(
    Number(cfg.sigma.maxCell),
    cfg.sigma.minCell + 0.001,
    0.49,
  );

  // Aniso warp controls
  cfg.anisoWarp.enabled = !!cfg.anisoWarp.enabled;
  cfg.anisoWarp.strength = clamp(Number(cfg.anisoWarp.strength), 0, 1.5);

  // Glass shard controls
  cfg.glassShard.enabled = !!cfg.glassShard.enabled;
  cfg.glassShard.mix = clamp(Number(cfg.glassShard.mix), 0, 1);
  cfg.glassShard.sharpness = clamp(Number(cfg.glassShard.sharpness), 0, 1);
  cfg.glassShard.edgeHardness = clamp(Number(cfg.glassShard.edgeHardness), 0, 1);
  cfg.glassShard.clipStrength = clamp(Number(cfg.glassShard.clipStrength), 0, 1);

  // Boolean toggles
  cfg.color.sparkleGradient = !!cfg.color.sparkleGradient;
  cfg.post.toneMapGamma = !!cfg.post.toneMapGamma;

  // Lighting vectors and colors
  cfg.lighting.direction = normalize3(cfg.lighting.direction);
  cfg.lighting.front.color = [
    Number(cfg.lighting.front.color?.[0] ?? 0.95),
    Number(cfg.lighting.front.color?.[1] ?? 0.68),
    Number(cfg.lighting.front.color?.[2] ?? 0.48),
  ];
  cfg.lighting.back.color = [
    Number(cfg.lighting.back.color?.[0] ?? 0.12),
    Number(cfg.lighting.back.color?.[1] ?? 0.16),
    Number(cfg.lighting.back.color?.[2] ?? 0.24),
  ];
  cfg.lighting.front.intensity = Math.max(
    0,
    Number(cfg.lighting.front.intensity),
  );
  cfg.lighting.back.intensity = Math.max(0, Number(cfg.lighting.back.intensity));

  // Render and motion helpers
  cfg.render.renderScale = clamp(Number(cfg.render.renderScale), 0.25, 2.0);

  cfg.motion.spinMeshes = !!cfg.motion.spinMeshes;
  cfg.motion.autoOrbit = !!cfg.motion.autoOrbit;
  cfg.motion.orbitSpeed = Number(cfg.motion.orbitSpeed) || 0;

  // Material flags
  cfg.material.backFaceCulling = !!cfg.material.backFaceCulling;

  return cfg;
}

/**
 * Convert the nested config shape into the original flat param shape used by the demo.
 * Useful for existing UIs that already bind to flat keys.
 */
export function centeredOctaveGaussianGlintConfigToLegacyParams(config) {
  const cfg = createCenteredOctaveGaussianGlintConfig(config);

  return {
    density01: cfg.glint.density01,
    roughness01: cfg.glint.roughness01,
    microfacetRoughness: cfg.glint.microfacetRoughness,
    pixelFilterSize: cfg.glint.pixelFilterSize,
    uvScale: cfg.glint.uvScale,
    intensity: cfg.glint.intensity,

    sigmaScale: cfg.sigma.scale,
    sigmaMinCell: cfg.sigma.minCell,
    sigmaMaxCell: cfg.sigma.maxCell,

    anisoWarpStrength: cfg.anisoWarp.strength,

    shardMix: cfg.glassShard.mix,
    shardSharpness: cfg.glassShard.sharpness,
    shardEdgeHardness: cfg.glassShard.edgeHardness,
    shardClipStrength: cfg.glassShard.clipStrength,

    renderScale: cfg.render.renderScale,
    orbitSpeed: cfg.motion.orbitSpeed,

    spinMeshes: cfg.motion.spinMeshes,
    autoOrbit: cfg.motion.autoOrbit,

    toneMapGamma: cfg.post.toneMapGamma,
    sparkleColorGradient: cfg.color.sparkleGradient,

    anisoWarpMode: cfg.anisoWarp.enabled,
    glassShardMode: cfg.glassShard.enabled,
  };
}

/**
 * Apply a legacy flat param object onto a nested config shape.
 * Returns a fresh validated config object.
 */
export function applyLegacyParamsToCenteredOctaveGaussianGlintConfig(
  params,
  config,
) {
  const cfg = config || createCenteredOctaveGaussianGlintConfig();

  cfg.glint.density01 = Number(params.density01);
  cfg.glint.roughness01 = Number(params.roughness01);
  cfg.glint.microfacetRoughness = Number(params.microfacetRoughness);
  cfg.glint.pixelFilterSize = Number(params.pixelFilterSize);
  cfg.glint.uvScale = Number(params.uvScale);
  cfg.glint.intensity = Number(params.intensity);

  cfg.sigma.scale = Number(params.sigmaScale);
  cfg.sigma.minCell = Number(params.sigmaMinCell);
  cfg.sigma.maxCell = Number(params.sigmaMaxCell);

  cfg.anisoWarp.strength = Number(params.anisoWarpStrength);
  cfg.anisoWarp.enabled = !!params.anisoWarpMode;

  cfg.glassShard.mix = Number(params.shardMix);
  cfg.glassShard.sharpness = Number(params.shardSharpness);
  cfg.glassShard.edgeHardness = Number(params.shardEdgeHardness);
  cfg.glassShard.clipStrength = Number(params.shardClipStrength);
  cfg.glassShard.enabled = !!params.glassShardMode;

  cfg.render.renderScale = Number(params.renderScale);

  cfg.motion.orbitSpeed = Number(params.orbitSpeed);
  cfg.motion.spinMeshes = !!params.spinMeshes;
  cfg.motion.autoOrbit = !!params.autoOrbit;

  cfg.post.toneMapGamma = !!params.toneMapGamma;
  cfg.color.sparkleGradient = !!params.sparkleColorGradient;

  return createCenteredOctaveGaussianGlintConfig(cfg);
}

/**
 * Vertex shader:
 * - passes world/local position and normals
 * - constructs a stable object-locked tangent basis from the local normal
 * - transforms tangent/bitangent to world space
 *
 * This avoids face orientation snapping on box faces compared to basis derived from unstable references.
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_VERTEX_GLSL = String.raw`
precision highp float;

attribute vec3 position;
attribute vec3 normal;

uniform mat4 world;
uniform mat4 worldViewProjection;

varying vec3 vPosW;
varying vec3 vPosL;
varying vec3 vNormalW;
varying vec3 vNormalL;
varying vec3 vTangentW;
varying vec3 vBitangentW;

vec3 safeNormalize3(vec3 v) {
  float l = length(v);
  return (l > 1e-8) ? (v / l) : vec3(1.0, 0.0, 0.0);
}

void buildLocalFaceBasis(vec3 nL, out vec3 tL, out vec3 bL) {
  vec3 n = safeNormalize3(nL);

  // Choose a stable reference axis that is not parallel to the normal.
  vec3 refAxis = (abs(n.y) > 0.999) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);

  // Gram-Schmidt tangent orthogonalization.
  tL = safeNormalize3(refAxis - n * dot(refAxis, n));
  bL = safeNormalize3(cross(n, tL));
}

void main(void) {
  vec4 worldPos = world * vec4(position, 1.0);

  vPosW = worldPos.xyz;
  vPosL = position;

  vec3 nL = safeNormalize3(normal);
  vNormalL = nL;
  vNormalW = safeNormalize3((world * vec4(nL, 0.0)).xyz);

  vec3 tL;
  vec3 bL;
  buildLocalFaceBasis(nL, tL, bL);

  vTangentW = safeNormalize3((world * vec4(tL, 0.0)).xyz);
  vBitangentW = safeNormalize3((world * vec4(bL, 0.0)).xyz);

  gl_Position = worldViewProjection * vec4(position, 1.0);
}
`;

/**
 * Fragment shader:
 * - triplanar projected domains in object space
 * - centered gaussian glints in sparse hashed cells across octaves
 * - hybrid neighbor search (1 cell / 2x2 / 3x3 when needed)
 * - optional anisotropic lens warp
 * - optional glass shard clipping mask
 * - colored sparkle mode with analytic palette
 * - simple tonemap + gamma toggle
 */
export const CENTERED_OCTAVE_GAUSSIAN_GLINT_FRAGMENT_GLSL = String.raw`
precision highp float;

varying vec3 vPosW;
varying vec3 vPosL;
varying vec3 vNormalW;
varying vec3 vNormalL;
varying vec3 vTangentW;
varying vec3 vBitangentW;

uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uFrontLightColor;
uniform float uFrontLightIntensity;
uniform vec3 uBackLightColor;
uniform float uBackLightIntensity;

uniform float uDensity01;
uniform float uRoughness01;
uniform float uMicrofacetRoughness;
uniform float uPixelFilterSize;
uniform float uUVScale;
uniform float uIntensity;

uniform float uSigmaScale;
uniform float uSigmaMinCell;
uniform float uSigmaMaxCell;

uniform float uAnisoWarpStrength;

uniform float uShardMix;
uniform float uShardSharpness;
uniform float uShardEdgeHardness;
uniform float uShardClipStrength;

uniform float uUseToneMapGamma;
uniform float uUseSparkleColorGradient;
uniform float uUseAnisoWarpMode;
uniform float uUseGlassShardMode;

const float pi = 3.14159265358979;

float satf(float x) {
  return clamp(x, 0.0, 1.0);
}

float det2(mat2 m) {
  return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

mat2 tr2(mat2 m) {
  return mat2(m[0][0], m[1][0], m[0][1], m[1][1]);
}

vec2 safeN2(vec2 v) {
  float l = length(v);
  return (l > 1e-8) ? v / l : vec2(1.0, 0.0);
}

vec2 lambert(vec3 v) {
  return v.xy / sqrt(1.0 + v.z);
}

// Maps a half-vector under GGX NDF to a unit disk parameterization and returns:
// xy = disk coordinates
// z  = jacobian determinant term for density correction
vec3 ndf_to_disk_ggx(vec3 v, float alpha) {
  vec3 hemi = vec3(v.xy / max(alpha, 1e-5), v.z);
  float denom = max(dot(hemi, hemi), 1e-8);
  vec2 v_disk = lambert(normalize(hemi)) * 0.5 + 0.5;
  float jacobian_determinant = 1.0 / (max(alpha * alpha, 1e-8) * denom * denom);
  return vec3(v_disk, jacobian_determinant);
}

mat2 inv_quadratic(mat2 M) {
  float D = det2(M);
  if (abs(D) < 1e-10) D = (D < 0.0) ? -1e-10 : 1e-10;

  float A = dot(M[0] / D, M[0] / D);
  float B = -dot(M[0] / D, M[1] / D);
  float C = dot(M[1] / D, M[1] / D);

  return mat2(C, B, B, A);
}

// Converts the derivative Jacobian into an ellipse basis for footprint estimation.
mat2 uv_ellipsoid(mat2 uv_J) {
  mat2 Q = inv_quadratic(tr2(uv_J));

  float tr = 0.5 * (Q[0][0] + Q[1][1]);
  float D = sqrt(max(0.0, tr * tr - det2(Q)));

  float l1 = max(tr - D, 1e-8);
  float l2 = max(tr + D, 1e-8);

  vec2 v1 = vec2(l1 - Q[1][1], Q[0][1]);
  vec2 v2 = vec2(Q[1][0], l2 - Q[0][0]);

  vec2 n = 1.0 / sqrt(vec2(l1, l2));
  return mat2(safeN2(v1) * n.x, safeN2(v2) * n.y);
}

// Hash helpers used for sparse cell activation and random sparkle parameters.
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 paletteLerp(float t, float a, float b, vec3 ca, vec3 cb) {
  float u = clamp((t - a) / max(b - a, 1e-6), 0.0, 1.0);
  return mix(ca, cb, u);
}

// Analytic rainbow-ish sparkle palette.
// This avoids texture sampling restrictions in divergent control flow on WebGPU.
vec3 sampleSparkleGradient(float t) {
  t = fract(t);

  vec3 c0 = vec3(1.00, 0.376, 0.376);
  vec3 c1 = vec3(1.00, 0.667, 0.353);
  vec3 c2 = vec3(1.00, 0.941, 0.510);
  vec3 c3 = vec3(0.588, 1.00, 0.863);
  vec3 c4 = vec3(0.471, 0.706, 1.00);
  vec3 c5 = vec3(0.824, 0.549, 1.00);
  vec3 c6 = vec3(1.00, 0.471, 0.784);
  vec3 c7 = vec3(1.00, 0.922, 0.961);

  if (t < 0.12) return paletteLerp(t, 0.00, 0.12, c0, c1);
  if (t < 0.26) return paletteLerp(t, 0.12, 0.26, c1, c2);
  if (t < 0.42) return paletteLerp(t, 0.26, 0.42, c2, c3);
  if (t < 0.58) return paletteLerp(t, 0.42, 0.58, c3, c4);
  if (t < 0.74) return paletteLerp(t, 0.58, 0.74, c4, c5);
  if (t < 0.88) return paletteLerp(t, 0.74, 0.88, c5, c6);
  return paletteLerp(t, 0.88, 1.00, c6, c7);
}

// Smith-like masking-shadowing helpers for GGX.
float G1_GGX(vec3 n, vec3 h, vec3 v, float alpha) {
  float ndotv = dot(n, v);
  if (ndotv < 0.0) return 0.0;

  float ndotv_sq = max(ndotv * ndotv, 1e-6);
  float tan_theta_sq = (1.0 - ndotv_sq) / ndotv_sq;
  float Gamma = -0.5 + 0.5 * sqrt(1.0 + alpha * alpha * tan_theta_sq);

  return 1.0 / (1.0 + Gamma);
}

float G_GGX(vec3 n, vec3 h, vec3 li, vec3 lo, float alpha) {
  return G1_GGX(n, h, li, alpha) * G1_GGX(n, h, lo, alpha);
}

// Multiply by transpose(m) without constructing transpose explicitly.
vec3 mulT(mat3 m, vec3 v) {
  return vec3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
}

// Triplanar blend weights from local normal.
// High exponent keeps seam blending narrow and preserves detail.
vec3 triplanarWeights(vec3 n) {
  vec3 an = abs(normalize(n));
  vec3 w = pow(an, vec3(8.0));
  float s = max(w.x + w.y + w.z, 1e-6);
  return w / s;
}

// Object-local triplanar projections and derivative Jacobians.
// The shader samples the glint domain in local object space to lock patterns to the mesh.
void sparkleDomainProjX(vec3 posL, float scale, out vec2 uv, out mat2 uvJ) {
  float s = max(scale, 1e-6);
  vec3 p = posL * s;
  vec3 dx = dFdx(p);
  vec3 dy = dFdy(p);

  uv = vec2(p.z, p.y);
  uvJ = mat2(vec2(dx.z, dx.y), vec2(dy.z, dy.y));
}

void sparkleDomainProjY(vec3 posL, float scale, out vec2 uv, out mat2 uvJ) {
  float s = max(scale, 1e-6);
  vec3 p = posL * s;
  vec3 dx = dFdx(p);
  vec3 dy = dFdy(p);

  uv = vec2(p.x, p.z);
  uvJ = mat2(vec2(dx.x, dx.z), vec2(dy.x, dy.z));
}

void sparkleDomainProjZ(vec3 posL, float scale, out vec2 uv, out mat2 uvJ) {
  float s = max(scale, 1e-6);
  vec3 p = posL * s;
  vec3 dx = dFdx(p);
  vec3 dy = dFdy(p);

  uv = vec2(p.x, p.y);
  uvJ = mat2(vec2(dx.x, dx.y), vec2(dy.x, dy.y));
}

// Softly clamps to a square boundary by radial scaling in Chebyshev norm.
// Used to keep warped samples inside the current cell.
vec2 clampSquareSoft(vec2 q, float limit) {
  float l = max(limit, 1e-4);
  vec2 a = abs(q);
  float m = max(a.x, a.y);
  if (m <= l) return q;
  return q * (l / max(m, 1e-6));
}

// Superellipse fill mask used as the base shard silhouette.
float superellipseFill(vec2 p, vec2 r, float powerN, float edgeSoft) {
  vec2 q = abs(p) / max(r, vec2(1e-5));
  float m = pow(q.x, powerN) + pow(q.y, powerN);
  return 1.0 - smoothstep(1.0 - edgeSoft, 1.0 + edgeSoft, m);
}

// Builds a randomized shard-like clip mask in cell-local coordinates.
float localShardClipMask(
  vec2 localCell,
  vec2 h0,
  vec2 h1,
  float shardMix,
  float shardSharp,
  float edgeHardness
) {
  vec2 axis = safeN2(h1 * 2.0 - 1.0);
  vec2 perp = vec2(-axis.y, axis.x);

  float ang = (h0.x - 0.5) * 1.2;
  float ca = cos(ang);
  float sa = sin(ang);
  mat2 R = mat2(ca, -sa, sa, ca);

  vec2 p = R * localCell;

  float longR = mix(0.14, 0.44, h0.y);
  float shortR = mix(0.035, 0.18, h1.x);

  float shardiness = mix(0.75, 2.2, shardSharp);
  longR *= mix(0.85, 1.35, shardMix);
  shortR *= mix(0.70, 1.10, shardMix);

  float powerN = mix(1.15, 0.58, shardiness);
  float edgeSoft = mix(0.18, 0.02, edgeHardness);

  float baseFill = superellipseFill(p, vec2(longR, shortR), powerN, edgeSoft);

  float cut1 = dot(p, safeN2(axis + 0.35 * perp)) - mix(0.02, 0.14, h0.x);
  float cut2 = dot(p, safeN2(-axis + 0.55 * perp)) - mix(0.00, 0.12, h1.y);

  float cutSoft = mix(0.16, 0.02, edgeHardness);
  float cutMask1 = 1.0 - smoothstep(-cutSoft, cutSoft, cut1);
  float cutMask2 = 1.0 - smoothstep(-cutSoft, cutSoft, cut2);

  float faceted = max(baseFill * cutMask1, baseFill * cutMask2);
  return mix(1.0, faceted, shardMix);
}

// Core multi-octave glint field evaluator.
// Returns:
// x   = scalar D-like density contribution
// yzw = average sparkle color of contributing cells
vec4 sparkleFieldOctaves(vec3 hLocal, float alpha, vec2 uv, mat2 uv_J, float anisoEval) {
  vec3 xa_d = ndf_to_disk_ggx(hLocal, alpha);
  vec2 x_a = xa_d.xy;
  float d = xa_d.z;

  float dens = satf(uDensity01);
  float rough = satf(uRoughness01);
  float micro = satf(uMicrofacetRoughness / 0.10);
  float fsz = max(uPixelFilterSize, 0.10);

  float baseRes = 12.0;
  float baseKeep = mix(0.02, 0.98, dens);

  float anisoStrength = anisoEval * satf(uAnisoWarpStrength / 1.5);

  float useShards = step(0.5, uUseGlassShardMode);
  float shardMix = satf(uShardMix) * useShards;
  float shardSharp = satf(uShardSharpness);
  float shardEdgeHard = satf(uShardEdgeHardness);
  float shardClipStrength = satf(uShardClipStrength);

  float sigmaScaleCtl = max(uSigmaScale, 0.01);
  float sigmaMinCtl = clamp(uSigmaMinCell, 0.005, 0.49);
  float sigmaMaxCtl = clamp(uSigmaMaxCell, sigmaMinCtl + 0.001, 0.49);

  vec3 Cacc = vec3(0.0);
  float Wacc = 0.0;
  float Dacc = 0.0;

  float useSparkle = step(0.5, uUseSparkleColorGradient);

  // Four octaves of sparse cell distributions.
  for (int k = 0; k < 4; ++k) {
    float fk = float(k);
    float res_s = baseRes * exp2(fk);

    vec2 p = uv * res_s;
    vec2 cellBase = floor(p);
    vec2 localCenter = fract(p) - 0.5;

    // Estimate local pixel footprint in cell units.
    mat2 Jcells = (res_s * fsz) * uv_J;
    float cellPerPx = max(length(Jcells[0]), length(Jcells[1]));

    // Sigma is stabilized across zoom by clamping to a footprint-dependent minimum.
    float sigmaBase = mix(0.078, 0.152, rough) * mix(0.95, 1.18, fk / 3.0);
    float sigmaMinPx = mix(0.28, 0.82, rough);
    float sigmaMinCells = sigmaMinPx * cellPerPx;

    float sigmaCellsBase = max(sigmaBase, sigmaMinCells);
    sigmaCellsBase *= sigmaScaleCtl;
    sigmaCellsBase = clamp(sigmaCellsBase, sigmaMinCtl, sigmaMaxCtl);

    float octaveKeep = baseKeep * mix(1.00, 0.72, fk / 3.0);
    float octaveWeight = exp2(-fk * 0.72);

    // Hybrid neighbor search:
    // - center cell always
    // - 2x2 near cell edges
    // - 3x3 for larger/grazing footprints and stronger warps
    float edgeBand = clamp(0.05 + 2.6 * sigmaCellsBase + 0.85 * cellPerPx, 0.05, 0.49);
    float edgeNear = step(0.5 - edgeBand, max(abs(localCenter.x), abs(localCenter.y)));

    float grazingNeed3x3 = 0.0;
    grazingNeed3x3 = max(grazingNeed3x3, step(0.22, cellPerPx));
    grazingNeed3x3 = max(grazingNeed3x3, step(0.155, sigmaCellsBase));
    grazingNeed3x3 = max(grazingNeed3x3, step(0.001, anisoStrength * 0.65));
    grazingNeed3x3 = max(grazingNeed3x3, step(0.001, shardMix * 0.65));

    int sx = (localCenter.x >= 0.0) ? 1 : -1;
    int sy = (localCenter.y >= 0.0) ? 1 : -1;

    for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        float allowThisCell = 0.0;

        if (ox == 0 && oy == 0) {
          allowThisCell = 1.0;
        }

        if (edgeNear > 0.5) {
          bool in2x2x = (ox == 0) || (ox == sx);
          bool in2x2y = (oy == 0) || (oy == sy);
          if (in2x2x && in2x2y) {
            allowThisCell = 1.0;
          }
        }

        if (grazingNeed3x3 > 0.5) {
          allowThisCell = 1.0;
        }

        if (allowThisCell < 0.5) {
          continue;
        }

        vec2 cellOffset = vec2(float(ox), float(oy));
        vec2 cell = cellBase + cellOffset;
        vec2 local = p - (cell + vec2(0.5));

        // Cheap cull in cell-local square before hash work.
        float localCullR = 0.78 + 2.8 * sigmaCellsBase + 0.9 * cellPerPx;
        if (max(abs(local.x), abs(local.y)) > localCullR) {
          continue;
        }

        // Deterministic per-cell random values.
        vec2 h0 = hash22(cell + vec2(17.1 * fk + 3.7, 29.3 * fk + 5.1));
        vec2 h1 = hash22(cell + vec2(41.9 * fk + 11.4, 13.7 * fk + 27.2));
        float h2 = hash12(cell + vec2(71.2 * fk + 9.0, 19.4 * fk + 2.0));

        float cellOn = step(1.0 - octaveKeep, h0.x);
        if (cellOn < 0.5) {
          continue;
        }

        float sigmaCells = sigmaCellsBase;

        vec2 axis = safeN2(h1 * 2.0 - 1.0);
        vec2 axisPerp = vec2(-axis.y, axis.x);

        vec2 q = local;

        // Optional anisotropic lens warp of the in-cell gaussian coordinates.
        if (anisoStrength > 0.001) {
          float stretch = 1.0 + anisoStrength * (0.40 + 1.10 * h0.y) * mix(0.6, 1.0, 1.0 - rough);
          float squeeze = mix(0.38, 0.90, rough) / max(stretch, 1e-4);

          float qx = dot(q, axis);
          float qy = dot(q, axisPerp);
          q = axis * (qx * stretch) + axisPerp * (qy * squeeze);

          vec2 lensVec = (x_a - vec2(0.5)) * (0.03 + 0.16 * anisoStrength) * mix(0.7, 1.25, h2);
          q += lensVec;

          float limit = clamp(0.49 - 2.0 * sigmaCells - 0.5 * cellPerPx, 0.03, 0.49);
          q = clampSquareSoft(q, limit);
        }

        // Surface-space gaussian in cell coordinates.
        float r2 = dot(q, q) / max(sigmaCells * sigmaCells, 1e-8);
        float gSurface = exp(-0.5 * r2);

        // Optional shard silhouette clip.
        if (shardMix > 0.001) {
          float localClip = localShardClipMask(local, h0, h1, shardMix, shardSharp, shardEdgeHard);
          float shardClip = mix(1.0, localClip, shardClipStrength);
          gSurface *= shardClip;
        }

        // Angular gaussian in GGX NDF disk domain.
        vec2 gA = 0.08 + 0.84 * h1;
        float sigmaA = mix(0.028, 0.095, micro) * mix(0.95, 1.18, rough);
        sigmaA = clamp(sigmaA, 0.018, 0.125);

        vec2 da = x_a - gA;
        float gAngular = exp(-0.5 * dot(da, da) / max(sigmaA * sigmaA, 1e-8));

        // Rare brighter sparkles improve perceived twinkle variation.
        float rarityBoost = mix(0.45, 4.20, pow(h0.y, 4.5));
        float contrib = cellOn * octaveWeight * rarityBoost * gSurface * gAngular;

        Dacc += contrib;

        // Optional analytic sparkle coloring.
        float t = fract(h2 + 0.61803398875 * h0.y + 0.27 * fk);
        vec3 c = sampleSparkleGradient(t);

        // Subtle shard tint bias to push a glassy look when shard mode is enabled.
        if (useShards > 0.5) {
          float cool = hash12(cell + vec2(3.11 * fk + 19.3, 5.77 * fk + 2.1));
          vec3 shardTint = mix(
            vec3(0.78, 0.90, 1.15),
            vec3(1.20, 0.95, 1.35),
            cool
          );
          c *= mix(vec3(1.0), shardTint, 0.28 * shardMix);
        }

        float cw = contrib * useSparkle;
        Cacc += c * cw;
        Wacc += cw;
      }
    }
  }

  float densityGain = mix(2.2, 5.8, pow(dens, 0.75));
  float D = Dacc * d / pi * densityGain;

  vec3 avgC = (Wacc > 1e-8) ? (Cacc / Wacc) : vec3(1.0);
  return vec4(D, avgC);
}

// Evaluates the colored glint BRDF term for one projected domain.
vec4 glintBRDFColored(float alpha, vec3 view, vec3 light, mat3 base, vec2 uv, mat2 uv_J, float anisoEval) {
  float ndotv = max(dot(base[2], view), 0.0);
  float ndotl = max(dot(base[2], light), 0.0);

  vec3 h_sum = view + light;
  float h2 = dot(h_sum, h_sum);
  float validH = step(1e-10, h2);

  vec3 h_world = h_sum * inversesqrt(max(h2, 1e-10));
  h_world = normalize(mix(vec3(0.0, 0.0, 1.0), h_world, validH));
  vec3 h_local = mulT(base, h_world);

  vec4 Dcol = sparkleFieldOctaves(h_local, alpha, uv, uv_J, anisoEval);

  float D = Dcol.x;

  // Fresnel is intentionally biased high for a strong glint look.
  float F = mix(pow(1.0 - dot(h_world, light), 5.0), 1.0, 0.96);

  float G = G_GGX(base[2], h_world, light, view, alpha);
  float denom = max(4.0 * ndotv * ndotl, 1e-6);

  float valid = step(1e-6, ndotv) * step(1e-6, ndotl) * validH;
  float spec = (D * F * G / denom) * valid;

  return vec4(spec, Dcol.yzw);
}

void main(void) {
  vec3 N = normalize(vNormalW);
  vec3 V = normalize(uCameraPos - vPosW);

  // Build triplanar projected domains in local object space.
  vec2 uvX;
  vec2 uvY;
  vec2 uvZ;
  mat2 uvJX;
  mat2 uvJY;
  mat2 uvJZ;

  sparkleDomainProjX(vPosL, uUVScale, uvX, uvJX);
  sparkleDomainProjY(vPosL, uUVScale, uvY, uvJY);
  sparkleDomainProjZ(vPosL, uUVScale, uvZ, uvJZ);

  uvJX = uv_ellipsoid(uvJX);
  uvJY = uv_ellipsoid(uvJY);
  uvJZ = uv_ellipsoid(uvJZ);

  vec3 triW = triplanarWeights(vNormalL);

  // Map user roughness [0,1] into the alpha range used by this shader.
  float alpha = 0.2 + uRoughness01 * 0.8;

  vec3 L0 = normalize(uLightDir);
  vec3 frontLColor = uFrontLightColor * uFrontLightIntensity;
  vec3 backLColor = uBackLightColor * uBackLightIntensity;

  float ndotlFront = max(dot(N, L0), 0.0);
  float ndotlBack = max(dot(N, -L0), 0.0);

  // Rebuild an orthonormal world basis from the stable tangent and normal.
  vec3 T = normalize(vTangentW - N * dot(N, vTangentW));
  if (length(T) < 1e-5) {
    vec3 refAxis = (abs(N.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(refAxis - N * dot(refAxis, N));
  }
  vec3 B = normalize(cross(N, T));
  mat3 glintBasis = mat3(T, B, N);

  float anisoEval = step(0.5, uUseAnisoWarpMode);

  // Evaluate both front and back light directions across all triplanar projections.
  vec4 sFrontX = glintBRDFColored(alpha, V,  L0, glintBasis, uvX, uvJX, anisoEval);
  vec4 sFrontY = glintBRDFColored(alpha, V,  L0, glintBasis, uvY, uvJY, anisoEval);
  vec4 sFrontZ = glintBRDFColored(alpha, V,  L0, glintBasis, uvZ, uvJZ, anisoEval);

  vec4 sBackX  = glintBRDFColored(alpha, V, -L0, glintBasis, uvX, uvJX, anisoEval);
  vec4 sBackY  = glintBRDFColored(alpha, V, -L0, glintBasis, uvY, uvJY, anisoEval);
  vec4 sBackZ  = glintBRDFColored(alpha, V, -L0, glintBasis, uvZ, uvJZ, anisoEval);

  vec4 sFrontPack = sFrontX * triW.x + sFrontY * triW.y + sFrontZ * triW.z;
  vec4 sBackPack  = sBackX  * triW.x + sBackY  * triW.y + sBackZ  * triW.z;

  float useSparkle = step(0.5, uUseSparkleColorGradient);
  vec3 sparkleFront = mix(vec3(1.0), sFrontPack.yzw, useSparkle);
  vec3 sparkleBack = mix(vec3(1.0), sBackPack.yzw, useSparkle);

  float intensityGain = uIntensity * 10.0;

  vec3 col =
    vec3(sFrontPack.x) * sparkleFront * ndotlFront * frontLColor * intensityGain +
    vec3(sBackPack.x)  * sparkleBack  * ndotlBack  * backLColor  * intensityGain;

  if (uUseToneMapGamma > 0.5) {
    col = col / (1.0 + col);
    col = pow(col, vec3(0.45454545));
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

/**
 * Installs the shader sources into Babylon's global shader store.
 * Returns the shader name actually used for installation.
 */
export function installCenteredOctaveGaussianGlintShader(opts = {}) {
  const shaderName = opts.shaderName || CENTERED_OCTAVE_GAUSSIAN_GLINT_SHADER_NAME;
  const vertexSource = opts.vertexGLSL || CENTERED_OCTAVE_GAUSSIAN_GLINT_VERTEX_GLSL;
  const fragmentSource =
    opts.fragmentGLSL || CENTERED_OCTAVE_GAUSSIAN_GLINT_FRAGMENT_GLSL;

  Effect.ShadersStore[shaderName + "VertexShader"] = vertexSource;
  Effect.ShadersStore[shaderName + "FragmentShader"] = fragmentSource;

  return shaderName;
}

/**
 * Creates and returns the ShaderMaterial plus the resolved config.
 *
 * Usage:
 *   const { material, config } = createCenteredOctaveGaussianGlintMaterial({ scene, config: overrides });
 */
export function createCenteredOctaveGaussianGlintMaterial({
  scene,
  config,
  shaderName,
  vertexGLSL,
  fragmentGLSL,
  materialName = "centeredOctaveGaussianGlintMaterial",
} = {}) {
  if (!scene) throw new Error("scene is required.");

  const cfg = createCenteredOctaveGaussianGlintConfig(config);

  const installedName = installCenteredOctaveGaussianGlintShader({
    shaderName: shaderName || cfg.shader.name,
    vertexGLSL,
    fragmentGLSL,
  });

  const material = new ShaderMaterial(
    materialName,
    scene,
    { vertex: installedName, fragment: installedName },
    {
      attributes: ["position", "normal"],
      uniforms: CENTERED_OCTAVE_GAUSSIAN_GLINT_UNIFORMS,
      shaderLanguage: ShaderLanguage.GLSL,
    },
  );

  material.backFaceCulling = !!cfg.material.backFaceCulling;

  return { material, shaderName: installedName, config: cfg };
}

/**
 * Uploads all glint uniforms for the current frame.
 * Call this each frame if camera position or config values change.
 *
 * Returns the validated config actually applied.
 */
export function applyCenteredOctaveGaussianGlintUniforms({
  material,
  camera,
  config,
} = {}) {
  if (!material) throw new Error("material is required.");
  if (!camera) throw new Error("camera is required.");

  const cfg = createCenteredOctaveGaussianGlintConfig(config);

  // Camera position drives view vector in the fragment shader.
  material.setVector3("uCameraPos", camera.globalPosition || camera.position);

  // Lighting
  material.setVector3(
    "uLightDir",
    toVector3(cfg.lighting.direction, [1, 1, 1], true),
  );

  material.setColor3(
    "uFrontLightColor",
    toColor3(cfg.lighting.front.color, [0.95, 0.68, 0.48]),
  );
  material.setFloat("uFrontLightIntensity", Number(cfg.lighting.front.intensity));

  material.setColor3(
    "uBackLightColor",
    toColor3(cfg.lighting.back.color, [0.12, 0.16, 0.24]),
  );
  material.setFloat("uBackLightIntensity", Number(cfg.lighting.back.intensity));

  // Glint field controls
  material.setFloat("uDensity01", Number(cfg.glint.density01));
  material.setFloat("uRoughness01", Number(cfg.glint.roughness01));
  material.setFloat("uMicrofacetRoughness", Number(cfg.glint.microfacetRoughness));
  material.setFloat("uPixelFilterSize", Number(cfg.glint.pixelFilterSize));
  material.setFloat("uUVScale", Number(cfg.glint.uvScale));
  material.setFloat("uIntensity", Number(cfg.glint.intensity));

  // Sigma controls
  material.setFloat("uSigmaScale", Number(cfg.sigma.scale));
  material.setFloat("uSigmaMinCell", Number(cfg.sigma.minCell));
  material.setFloat("uSigmaMaxCell", Number(cfg.sigma.maxCell));

  // Aniso warp
  material.setFloat("uAnisoWarpStrength", Number(cfg.anisoWarp.strength));

  // Glass shard controls
  material.setFloat("uShardMix", Number(cfg.glassShard.mix));
  material.setFloat("uShardSharpness", Number(cfg.glassShard.sharpness));
  material.setFloat("uShardEdgeHardness", Number(cfg.glassShard.edgeHardness));
  material.setFloat("uShardClipStrength", Number(cfg.glassShard.clipStrength));

  // Feature toggles
  material.setFloat("uUseToneMapGamma", cfg.post.toneMapGamma ? 1.0 : 0.0);
  material.setFloat(
    "uUseSparkleColorGradient",
    cfg.color.sparkleGradient ? 1.0 : 0.0,
  );
  material.setFloat("uUseAnisoWarpMode", cfg.anisoWarp.enabled ? 1.0 : 0.0);
  material.setFloat("uUseGlassShardMode", cfg.glassShard.enabled ? 1.0 : 0.0);

  return cfg;
}

/**
 * Applies the configured renderScale to a Babylon engine as hardware scaling level.
 * This is a convenience helper that matches the original demo behavior.
 *
 * Note:
 * Babylon's setHardwareScalingLevel uses larger numbers for lower internal resolution.
 * This helper simply mirrors the original control semantics from your demo code.
 */
export function applyCenteredOctaveGaussianGlintRenderScale(engine, config) {
  if (!engine) throw new Error("engine is required.");
  const cfg = createCenteredOctaveGaussianGlintConfig(config);
  engine.setHardwareScalingLevel(Number(cfg.render.renderScale));
  engine.resize();
  return cfg.render.renderScale;
}

/*
// Example imports for your ESM demo page or app module
import {
  CENTERED_OCTAVE_GAUSSIAN_GLINT_LEGACY_DEFAULTS,
  createCenteredOctaveGaussianGlintConfig,
  centeredOctaveGaussianGlintConfigToLegacyParams,
  applyLegacyParamsToCenteredOctaveGaussianGlintConfig,
  createCenteredOctaveGaussianGlintMaterial,
  applyCenteredOctaveGaussianGlintUniforms,
  applyCenteredOctaveGaussianGlintRenderScale,
} from "./glint/centeredOctaveGaussianGlint.js";
*/