#version 450

layout(location = 0) in vec2 inPos;       // Pixel coordinates (0..screenW, 0..screenH)
layout(location = 1) in vec2 inUV;        // Texture UV or local SDF coordinates
layout(location = 2) in vec4 inColor;     // Primary RGBA color
layout(location = 3) in vec4 inRect;      // (centerX, centerY, halfW, halfH) for SDF shapes
layout(location = 4) in vec4 inParams;    // x = mode, y = cornerRadius, z = borderThickness, w = texIndex

layout(push_constant) uniform UiPushConstants {
    vec2 screenSize;
    vec2 pad;
} pc;

layout(location = 0) out vec2 fragPixelPos;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec4 fragColor;
layout(location = 3) out vec4 fragRect;
layout(location = 4) out vec4 fragParams;

void main() {
    vec2 ndc = (inPos / pc.screenSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);
    fragPixelPos = inPos;
    fragUV = inUV;
    fragColor = inColor;
    fragRect = inRect;
    fragParams = inParams;
}
