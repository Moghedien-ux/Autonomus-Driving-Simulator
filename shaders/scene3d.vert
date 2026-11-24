#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in float inMatType;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 camPosAndTime; // xyz = camera world pos, w = sim_time
} pc;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragColor;
layout(location = 3) out vec2 fragUV;
layout(location = 4) out float fragMatType;

void main() {
    gl_Position = pc.viewProj * vec4(inPos, 1.0);
    fragWorldPos = inPos;
    fragNormal = normalize(inNormal);
    fragColor = inColor;
    fragUV = inUV;
    fragMatType = inMatType;
}
