#version 450

layout(location = 0) in vec2 fragPixelPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec4 fragColor;
layout(location = 3) in vec4 fragRect;   // (centerX, centerY, halfW, halfH)
layout(location = 4) in vec4 fragParams; // x = mode, y = radius, z = borderW, w = texIdx

layout(set = 0, binding = 0) uniform sampler2D uTextures[9];

layout(location = 0) out vec4 outColor;

float sdRoundedBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

// Safe compile-time constant sampler array lookup (avoids non-uniform indexing artifacts on NVIDIA/Intel Vulkan)
vec4 sampleTextureSlot(int tid, vec2 uv) {
    if (tid == 0) return texture(uTextures[0], uv);
    if (tid == 1) return texture(uTextures[1], uv);
    if (tid == 2) return texture(uTextures[2], uv);
    if (tid == 3) return texture(uTextures[3], uv);
    if (tid == 4) return texture(uTextures[4], uv);
    if (tid == 5) return texture(uTextures[5], uv);
    if (tid == 6) return texture(uTextures[6], uv);
    if (tid == 7) return texture(uTextures[7], uv);
    return texture(uTextures[8], uv);
}

void main() {
    int mode = int(fragParams.x + 0.5);

    // Mode 0: Solid / Gradient Flat Primitive
    if (mode == 0) {
        outColor = fragColor;
        return;
    }

    // Mode 1: Analytical Per-Pixel SDF Rounded Card / Pill (Fill or Border)
    if (mode == 1) {
        vec2 p = fragPixelPos - fragRect.xy;
        float r = min(fragParams.y, min(fragRect.z, fragRect.w));
        float d = sdRoundedBox(p, fragRect.zw, r);
        float borderW = fragParams.z;

        if (borderW <= 0.0) {
            float alpha = 1.0 - smoothstep(-0.65, 0.65, d);
            outColor = vec4(fragColor.rgb, fragColor.a * alpha);
        } else {
            float outer = 1.0 - smoothstep(-0.65, 0.65, d);
            float inner = 1.0 - smoothstep(-0.65, 0.65, d + borderW);
            float ringAlpha = clamp(outer - inner, 0.0, 1.0);
            outColor = vec4(fragColor.rgb, fragColor.a * ringAlpha);
        }
        return;
    }

    // Mode 2: Soft SDF Drop Shadow behind Cards
    if (mode == 2) {
        vec2 p = fragPixelPos - fragRect.xy;
        float r = min(fragParams.y, min(fragRect.z, fragRect.w));
        float d = sdRoundedBox(p, fragRect.zw, r);
        float shadow = 1.0 - smoothstep(-2.0, fragParams.z, d);
        outColor = vec4(fragColor.rgb, fragColor.a * shadow * shadow);
        return;
    }

    // Mode 3: Anti-Aliased Font Atlas Glyph (uTextures[0])
    if (mode == 3) {
        float cov = texture(uTextures[0], fragUV).r;
        outColor = vec4(fragColor.rgb, fragColor.a * cov);
        return;
    }

    // Mode 4: Rounded Image / Bilinear Texture Widget (Saliency Maps & Conv2D Feature Maps)
    if (mode == 4) {
        int tid = clamp(int(fragParams.w + 0.5), 0, 8);
        vec4 texCol = sampleTextureSlot(tid, fragUV);
        vec2 p = fragPixelPos - fragRect.xy;
        float r = min(fragParams.y, min(fragRect.z, fragRect.w));
        float d = sdRoundedBox(p, fragRect.zw, r);
        float mask = 1.0 - smoothstep(-0.65, 0.65, d);
        outColor = vec4(texCol.rgb * fragColor.rgb, texCol.a * fragColor.a * mask);
        return;
    }

    // Mode 5: Analytical Anti-Aliased Circular Ring / Arc
    if (mode == 5) {
        vec2 p = fragPixelPos - fragRect.xy;
        float dist = abs(length(p) - fragParams.y);
        float halfThick = fragParams.z * 0.5;
        float alpha = 1.0 - smoothstep(halfThick - 0.65, halfThick + 0.65, dist);
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
        return;
    }

    outColor = fragColor;
}
