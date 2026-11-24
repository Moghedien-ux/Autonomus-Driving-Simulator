#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragColor;
layout(location = 3) in vec2 fragUV;
layout(location = 4) in float fragMatType;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 camPosAndTime;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    int mat = int(fragMatType + 0.5);

    // Mat 6: 3D Sky Horizon Dome (smooth vertical atmospheric gradient)
    if (mat == 6) {
        float h = clamp(fragWorldPos.z / 95.0, 0.0, 1.0);
        vec3 horizonCol = vec3(0.18, 0.24, 0.34);
        vec3 zenithCol  = vec3(0.07, 0.10, 0.17);
        vec3 sky = mix(horizonCol, zenithCol, pow(h, 0.65));
        outColor = vec4(sky, 1.0);
        return;
    }

    // Mat 2: unlit / emissive (signals, lights, road markings, trajectory overlay)
    if (mat == 2) {
        outColor = fragColor;
        return;
    }

    // Mat 3: Soft radial/rounded Ground Ambient Occlusion Shadow
    if (mat == 3) {
        vec2 d = abs(fragUV) - vec2(0.55, 0.35);
        float dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
        float alpha = fragColor.a * (1.0 - smoothstep(-0.15, 0.45, dist));
        outColor = vec4(fragColor.rgb, alpha);
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(pc.camPosAndTime.xyz - fragWorldPos);
    vec3 L = normalize(vec3(-0.40, -0.52, 0.75)); // Bright, crisp sun angle
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float skyHemi = 0.5 + 0.5 * N.z;
    vec3 ambient = mix(vec3(0.28, 0.31, 0.37), vec3(0.54, 0.60, 0.70), skyHemi);
    vec3 diffuse = vec3(1.00, 0.98, 0.94) * (0.65 * NdotL);

    vec3 baseCol = fragColor.rgb;

    // Mat 4: Procedural Asphalt Road Surface
    if (mat == 4) {
        float grain = fract(sin(dot(floor(fragWorldPos.xy * 12.0), vec2(12.9898, 78.233))) * 43758.5453);
        baseCol += (grain - 0.5) * 0.015;
        float specRoad = pow(max(dot(N, H), 0.0), 20.0) * 0.06;
        vec3 litRoad = baseCol * (ambient + diffuse) + vec3(specRoad);
        outColor = vec4(clamp(litRoad, 0.0, 1.0), fragColor.a);
        return;
    }

    vec3 lit = baseCol * (ambient + diffuse);

    // Mat 1: Metallic Automotive Clearcoat Paint & Tinted Cabin Glass
    if (mat == 1) {
        float NdotV = max(dot(N, V), 0.0);
        float fresnel = pow(1.0 - NdotV, 3.2);
        float spec = pow(max(dot(N, H), 0.0), 56.0) * 0.70;
        vec3 skyReflect = mix(vec3(0.22, 0.27, 0.36), vec3(0.62, 0.76, 0.96), clamp(reflect(-V, N).z * 0.5 + 0.5, 0.0, 1.0));
        lit += vec3(spec) + skyReflect * (fresnel * 0.36);
    } else if (mat == 5) {
        // Mat 5: Architectural Building Facade with crisp glass window grid & warm lit interiors
        if (abs(N.z) < 0.2 && fragWorldPos.z > 2.2) {
            float horizCoord = (abs(N.x) > 0.5) ? fragWorldPos.y : fragWorldPos.x;
            float winX = fract(horizCoord * 0.42);
            float winZ = fract(fragWorldPos.z * 0.34);
            bool inWindow = (winX > 0.14 && winX < 0.86 && winZ > 0.18 && winZ < 0.82);
            if (inWindow) {
                float cellHash = fract(sin(floor(horizCoord * 0.42) * 19.1 + floor(fragWorldPos.z * 0.34) * 47.3) * 43758.5);
                vec3 glassCol = mix(vec3(0.14, 0.21, 0.31), vec3(0.30, 0.42, 0.58), skyHemi);
                if (cellHash > 0.65) {
                    glassCol = mix(glassCol, vec3(0.85, 0.74, 0.52), 0.48); // Warm lit office windows
                }
                float specGlass = pow(max(dot(N, H), 0.0), 48.0) * 0.50;
                lit = glassCol * (0.72 + 0.42 * NdotL) + vec3(specGlass);
            }
        }
    }

    float distToCam = length(pc.camPosAndTime.xyz - fragWorldPos);
    float fogFactor = clamp((distToCam - 110.0) / 240.0, 0.0, 0.45);
    vec3 fogColor = vec3(0.14, 0.19, 0.27);
    lit = mix(lit, fogColor, fogFactor);

    outColor = vec4(clamp(lit, 0.0, 1.0), fragColor.a);
}
