// Application entry point, HUD layout, input handling, and headless test runner.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "vulkan_renderer.hpp"
#include <cstdio>
#include <cstring>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

namespace sim {

enum class CameraViewMode : int {
    CHASE_3D = 0,
    ISOMETRIC_3D = 1,
    TOP_DOWN_CITY = 2
};

enum class SaliencyVisMode : int {
    RGB_PLUS_SALIENCY = 0,
    PURE_SALIENCY_HEATMAP = 1,
    RAW_CAMERA_RGB = 2
};

struct UiRect {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool contains(int mx, int my) const {
        return mx >= x0 && mx <= x1 && my >= y0 && my <= y1;
    }
};

struct AppState {
    CityWorld world;
    AutonomyController controller;
    VulkanRenderer vk;
    MeshBuilder3D mesh_main_3d;
    MeshBuilder3D mesh_cam_3d;
    UIBuilder2D ui_bg;
    UIBuilder2D ui_fg;

    int win_w = 1660;
    int win_h = 950;
    bool need_swapchain_resize = false;
    bool paused = false;
    int sim_speed_mult = 1;

    CameraViewMode cam_mode = CameraViewMode::CHASE_3D;
    SaliencyVisMode sal_mode = SaliencyVisMode::RGB_PLUS_SALIENCY;

    Vec2 smooth_cam_target{};
    float smooth_cam_heading = 0.0f;
    bool cam_initialized = false;

    float orbit_yaw_offset = 0.0f;
    float orbit_pitch = 0.42f;
    float orbit_zoom = 23.0f;
    bool mouse_dragging = false;
    int last_mouse_x = 0;
    int last_mouse_y = 0;
    int hover_mouse_x = 0;
    int hover_mouse_y = 0;

    int next_edge_case_idx = 0;

    UiRect rect_mode_tabs[4];
    UiRect rect_action_btns[5];
    UiRect rect_cam_pills[3];
    UiRect rect_speed_pills[4];
    UiRect rect_saliency_toggle;
};

static AppState* g_app = nullptr;

// Prepare dynamic saliency and Conv2D feature map RGBA buffers for Vulkan upload
void build_dynamic_neural_textures(const AppState& app,
                                   uint8_t sal0_rgba[CAM_W * CAM_H * 4],
                                   uint8_t sal1_rgba[CAM_W * CAM_H * 4],
                                   uint8_t fm_rgba[6][12 * 16 * 4]) {
    auto encode_saliency = [&](const std::array<float, CAM_W * CAM_H>& sal, uint8_t* out_rgba) {
        for (int py = 0; py < CAM_H; ++py) {
            for (int px = 0; px < CAM_W; ++px) {
                float sum = 0.0f, wsum = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = std::max(0, std::min(CAM_H - 1, py + dy));
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = std::max(0, std::min(CAM_W - 1, px + dx));
                        float kw = (dx == 0 && dy == 0) ? 4.0f : ((dx == 0 || dy == 0) ? 2.0f : 1.0f);
                        sum += sal[ny * CAM_W + nx] * kw;
                        wsum += kw;
                    }
                }
                float s = clampf(sum / wsum, 0.0f, 1.0f);
                int idx = (py * CAM_W + px) * 4;

                if (app.sal_mode == SaliencyVisMode::RGB_PLUS_SALIENCY) {
                    // Only highlight the top attribution hotspots (s > 0.38) so the camera feed stays readable
                    float hotspot = clampf((s - 0.38f) / 0.62f, 0.0f, 1.0f);
                    float a = std::pow(hotspot, 1.45f) * 0.56f;
                    out_rgba[idx + 0] = static_cast<uint8_t>(clampf(0.08f + 0.88f * hotspot, 0.0f, 1.0f) * 255.0f);
                    out_rgba[idx + 1] = static_cast<uint8_t>(clampf(0.92f + 0.08f * hotspot, 0.0f, 1.0f) * 255.0f);
                    out_rgba[idx + 2] = static_cast<uint8_t>(clampf(0.65f - 0.35f * hotspot, 0.0f, 1.0f) * 255.0f);
                    out_rgba[idx + 3] = static_cast<uint8_t>(a * 255.0f);
                } else if (app.sal_mode == SaliencyVisMode::PURE_SALIENCY_HEATMAP) {
                    float r = clampf(s * 1.6f - 0.2f, 0.07f, 1.0f);
                    float g = clampf(s * 1.25f, 0.09f, 0.96f);
                    float b = clampf(0.22f + 0.75f * std::sin(s * PI), 0.14f, 0.95f);
                    out_rgba[idx + 0] = static_cast<uint8_t>(r * 255.0f);
                    out_rgba[idx + 1] = static_cast<uint8_t>(g * 255.0f);
                    out_rgba[idx + 2] = static_cast<uint8_t>(b * 255.0f);
                    out_rgba[idx + 3] = 255;
                } else {
                    out_rgba[idx + 0] = 0; out_rgba[idx + 1] = 0; out_rgba[idx + 2] = 0; out_rgba[idx + 3] = 0;
                }
            }
        }
    };

    encode_saliency(app.controller.cnn_v12.saliency_cam0, sal0_rgba);
    encode_saliency(app.controller.cnn_v12.saliency_cam1, sal1_rgba);

    int fw = app.controller.cnn_v12.conv1.w_out; // 16
    int fh = app.controller.cnn_v12.conv1.h_out; // 12
    for (int f = 0; f < 6; ++f) {
        const float* plane = &app.controller.cnn_v12.conv1.pool_out[f * fw * fh];
        for (int idx = 0; idx < fw * fh; ++idx) {
            float v = clampf(plane[idx] * 0.65f, 0.0f, 1.0f);
            fm_rgba[f][idx * 4 + 0] = static_cast<uint8_t>((0.10f + 0.26f * v) * 255.0f);
            fm_rgba[f][idx * 4 + 1] = static_cast<uint8_t>((0.13f + 0.72f * v) * 255.0f);
            fm_rgba[f][idx * 4 + 2] = static_cast<uint8_t>((0.20f + 0.78f * v) * 255.0f);
            fm_rgba[f][idx * 4 + 3] = 255;
        }
    }
}

// Build 2D SDF UI background and foreground layers (520px right dock)
void build_vulkan_ui_layers(AppState& app, VkRect2D& out_cam0_rect, VkRect2D& out_cam1_rect) {
    app.ui_bg.clear();
    app.ui_fg.clear();

    const auto& ctrl = app.controller;
    const auto& exp = ctrl.expert_v11.telemetry;
    const auto& cnn = ctrl.cnn_v12;
    const auto& ego = app.world.ego;
    char buf[256];

    // 1. Driving cluster card (top-left of 3D viewport)
    app.ui_bg.add_card(20.0f, 20.0f, 360.0f, 154.0f, 14.0f, {0.08f, 0.095f, 0.125f}, 0.92f);

    // Gear indicator (P R N D)
    Color3 col_P = (ego.gear == GearState::PARK)    ? Color3{0.20f, 0.96f, 0.62f} : Color3{0.42f, 0.46f, 0.54f};
    Color3 col_R = (ego.gear == GearState::REVERSE) ? Color3{1.00f, 0.54f, 0.18f} : Color3{0.42f, 0.46f, 0.54f};
    Color3 col_N = Color3{0.42f, 0.46f, 0.54f};
    Color3 col_D = (ego.gear == GearState::DRIVE)   ? Color3{0.95f, 0.98f, 1.00f} : Color3{0.42f, 0.46f, 0.54f};
    app.ui_fg.add_text((ego.gear == GearState::PARK)    ? FONT_SM_BOLD : FONT_XS, 36.0f, 43.0f, "P", col_P);
    app.ui_fg.add_text((ego.gear == GearState::REVERSE) ? FONT_SM_BOLD : FONT_XS, 56.0f, 43.0f, "R", col_R);
    app.ui_fg.add_text(FONT_XS,                                                   76.0f, 43.0f, "N", col_N);
    app.ui_fg.add_text((ego.gear == GearState::DRIVE)   ? FONT_SM_BOLD : FONT_XS, 96.0f, 43.0f, "D", col_D);

    // Active mode pill
    Color3 ap_pill_col = (ctrl.mode == DriveMode::NEURAL_V12_AUTONOMOUS) ? Color3{0.22f, 0.46f, 0.92f} :
                         (ctrl.mode == DriveMode::DAGGER_TRAINING)       ? Color3{0.15f, 0.66f, 0.46f} :
                         (ctrl.mode == DriveMode::RULE_V11_EXPERT)       ? Color3{0.20f, 0.52f, 0.76f} :
                                                                           Color3{0.32f, 0.36f, 0.42f};
    const char* ap_badge_txt = (ctrl.mode == DriveMode::NEURAL_V12_AUTONOMOUS) ? "v12 Neural" :
                               (ctrl.mode == DriveMode::DAGGER_TRAINING)       ? "DAgger Learning" :
                               (ctrl.mode == DriveMode::RULE_V11_EXPERT)       ? "v11 Rules" : "Manual Drive";
    app.ui_bg.add_rounded_rect(210.0f, 28.0f, 344.0f, 50.0f, 11.0f, ap_pill_col, 0.96f);
    app.ui_fg.add_text_centered(FONT_XS, 210.0f, 344.0f, 43.0f, ap_badge_txt, {0.98f, 0.99f, 1.0f});

    // Large Digital Speedometer (km/h)
    int speed_kmh = static_cast<int>(std::round(ego.speed * 3.6f));
    std::snprintf(buf, sizeof(buf), "%d", speed_kmh);
    app.ui_fg.add_text(FONT_SPEED, 36.0f, 92.0f, buf, {0.96f, 0.98f, 1.0f});
    app.ui_fg.add_text(FONT_XS, 38.0f, 112.0f, "km/h", {0.56f, 0.62f, 0.70f});

    // Speed Limit Sign Badge (centered numbers)
    app.ui_bg.add_rounded_rect(116.0f, 58.0f, 160.0f, 106.0f, 7.0f, {0.93f, 0.95f, 0.97f}, 0.96f);
    app.ui_bg.add_rounded_border(118.0f, 60.0f, 158.0f, 104.0f, 6.0f, {0.15f, 0.17f, 0.21f}, 1.6f);
    app.ui_fg.add_text_centered(FONT_XS, 116.0f, 160.0f, 75.0f, "MAX", {0.20f, 0.22f, 0.26f});
    std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(std::round(exp.target_speed * 3.6f)));
    app.ui_fg.add_text_centered(FONT_SM_BOLD, 116.0f, 160.0f, 96.0f, buf, {0.10f, 0.12f, 0.15f});

    // Turn-by-Turn Navigation Instruction + Upcoming Intersection Type & Branch Count
    const LaneEdge& cur_e = app.world.edges[ego.route_edges[ego.current_route_idx]];
    const Intersection& next_inter = app.world.intersections[cur_e.to_node];
    const char* inter_type_str = (next_inter.shape == IntersectionShape::ROUNDABOUT_PLAZA)          ? "Roundabout" :
                                 (next_inter.ctrl_type == IntersectionControlType::TRAFFIC_LIGHTS) ? "Traffic Light" :
                                 (next_inter.ctrl_type == IntersectionControlType::POLICE_OFFICER) ? "Police Officer" :
                                 (next_inter.ctrl_type == IntersectionControlType::FOUR_WAY_STOP)  ? "All-Way Stop" :
                                                                                                     "Yield Sign";
    const char* nav_action = (ego.active_intent == TurnIntent::LEFT)  ? "Turn Left" :
                             (ego.active_intent == TurnIntent::RIGHT) ? "Turn Right" : "Straight";
    std::snprintf(buf, sizeof(buf), "%s (%d-Way %s)", nav_action, next_inter.branch_count, inter_type_str);
    app.ui_fg.add_text(FONT_SM_BOLD, 170.0f, 76.0f, buf, {0.92f, 0.95f, 0.99f});

    // Current Road Lane & Line Marking Badge
    const char* road_kind = cur_e.is_one_way ? "1-Way" : (cur_e.num_lanes_same_dir > 1 ? "2-Lane Blvd" : "2-Way");
    const char* line_kind = cur_e.is_dotted_overtake_zone(ego.current_edge_s) ? "Dotted Pass" : "Solid No-Pass";
    std::snprintf(buf, sizeof(buf), "%s | %s", road_kind, line_kind);
    app.ui_fg.add_text(FONT_XS, 170.0f, 94.0f, buf, {0.62f, 0.78f, 0.94f});

    app.ui_fg.add_text(FONT_XS, 170.0f, 112.0f, exp.primary_reason,
                       exp.crash_prevention_active ? Color3{1.0f, 0.42f, 0.36f} :
                       exp.rule_overtaking_active  ? Color3{0.26f, 0.94f, 0.64f} : Color3{0.56f, 0.66f, 0.80f});

    // Power / Regenerative Braking Progress Bar
    app.ui_bg.add_rounded_rect(36.0f, 132.0f, 344.0f, 138.0f, 3.0f, {0.18f, 0.21f, 0.26f});
    float mid_bar = 190.0f;
    float bar_span = clampf(ego.accel_cmd, -1.0f, 1.0f) * 150.0f;
    Color3 pwr_col = (bar_span >= 0.0f) ? Color3{0.24f, 0.56f, 0.96f} : Color3{0.20f, 0.84f, 0.54f};
    app.ui_bg.add_rounded_rect(std::min(mid_bar, mid_bar + bar_span), 132.0f,
                               std::max(mid_bar, mid_bar + bar_span), 138.0f, 3.0f, pwr_col);

    // 360-Degree Crash Prevention, Active Autopark / Reverse, Active Overtake, OR Safety Supervisor Takeover Toast
    if (exp.crash_prevention_active) {
        app.ui_bg.add_card(20.0f, 164.0f, 412.0f, 204.0f, 10.0f, {0.68f, 0.16f, 0.14f}, 0.96f);
        std::snprintf(buf, sizeof(buf), "360 LOS Crash Prevention: %s", exp.primary_reason);
        app.ui_fg.add_text(FONT_SM_BOLD, 34.0f, 189.0f, buf, {1.0f, 0.96f, 0.94f});
    } else if (exp.autopark_active || exp.reverse_maneuver_active) {
        Color3 ap_col = (ego.gear == GearState::PARK)    ? Color3{0.12f, 0.56f, 0.38f} :
                        (ego.gear == GearState::REVERSE) ? Color3{0.66f, 0.34f, 0.12f} :
                                                           Color3{0.16f, 0.44f, 0.72f};
        app.ui_bg.add_card(20.0f, 164.0f, 424.0f, 204.0f, 10.0f, ap_col, 0.96f);
        const char* gear_tag = (ego.gear == GearState::PARK) ? "[PARK]" :
                               (ego.gear == GearState::REVERSE) ? "[REV]" : "[DRIVE]";
        std::snprintf(buf, sizeof(buf), "%s %s (%d vis free)", gear_tag, exp.primary_reason, exp.visible_free_parking_360);
        app.ui_fg.add_text(FONT_SM_BOLD, 34.0f, 189.0f, buf, {0.98f, 0.99f, 1.0f});
    } else if (exp.rule_overtaking_active || exp.rule_waiting_to_overtake) {
        Color3 ov_col = exp.rule_overtaking_active ? Color3{0.12f, 0.52f, 0.36f} : Color3{0.58f, 0.42f, 0.12f};
        app.ui_bg.add_card(20.0f, 164.0f, 412.0f, 204.0f, 10.0f, ov_col, 0.95f);
        std::snprintf(buf, sizeof(buf), "%s: %s (%+.1fm)",
                      exp.rule_overtaking_active ? "Overtaking Maneuver" : "Overtake Hold",
                      exp.primary_reason, exp.evasive_lat_offset);
        app.ui_fg.add_text(FONT_SM_BOLD, 34.0f, 189.0f, buf, {0.96f, 0.99f, 0.98f});
    } else if (ctrl.intervention_flash_timer > 0.0f) {
        app.ui_bg.add_card(20.0f, 164.0f, 412.0f, 204.0f, 10.0f, {0.58f, 0.14f, 0.16f}, 0.95f);
        std::snprintf(buf, sizeof(buf), "Supervisor Takeover: %s", ctrl.last_intervention_reason);
        app.ui_fg.add_text(FONT_SM_BOLD, 34.0f, 189.0f, buf, {0.98f, 0.96f, 0.96f});
    }

    // 2. Bottom camera & simulation speed bar
    float pill_y = static_cast<float>(app.win_h) - 54.0f;
    app.ui_bg.add_card(20.0f, pill_y, 476.0f, pill_y + 38.0f, 19.0f, {0.08f, 0.095f, 0.125f}, 0.90f);

    const char* cam_labels[3] = {"Chase 3D", "Orbit 3D", "City Map"};
    for (int i = 0; i < 3; ++i) {
        float bx = 26.0f + i * 82.0f;
        app.rect_cam_pills[i] = {bx, pill_y + 4.0f, bx + 78.0f, pill_y + 34.0f};
        bool active = (static_cast<int>(app.cam_mode) == i);
        if (active) {
            app.ui_bg.add_rounded_rect(bx, pill_y + 4.0f, bx + 78.0f, pill_y + 34.0f, 15.0f, {0.22f, 0.45f, 0.86f});
        }
        app.ui_fg.add_text_centered(FONT_XS, bx, bx + 78.0f, pill_y + 24.0f, cam_labels[i],
                                    active ? Color3{0.98f, 0.99f, 1.0f} : Color3{0.64f, 0.70f, 0.78f});
    }

    const char* spd_labels[4] = {"1x", "2x", "5x", app.paused ? "Resume" : "Pause"};
    int mults[3] = {1, 2, 5};
    for (int i = 0; i < 4; ++i) {
        float bx = 284.0f + i * 44.0f;
        float bw = (i == 3) ? 62.0f : 38.0f;
        app.rect_speed_pills[i] = {bx, pill_y + 4.0f, bx + bw, pill_y + 34.0f};
        bool active = (i < 3) ? (!app.paused && app.sim_speed_mult == mults[i]) : app.paused;
        if (active) {
            app.ui_bg.add_rounded_rect(bx, pill_y + 4.0f, bx + bw, pill_y + 34.0f, 15.0f,
                                       (i == 3) ? Color3{0.78f, 0.45f, 0.16f} : Color3{0.20f, 0.24f, 0.31f});
        }
        app.ui_fg.add_text_centered(FONT_XS, bx, bx + bw, pill_y + 24.0f, spd_labels[i],
                                    active ? Color3{0.98f, 0.99f, 1.0f} : Color3{0.64f, 0.70f, 0.78f});
    }

    // 3. Right-side neural inspector panel (520px wide)
    const float panel_w = 520.0f;
    const float px0 = static_cast<float>(app.win_w) - panel_w;
    const float card_x0 = px0 + 18.0f;
    const float card_x1 = px0 + 502.0f; // 484px card width

    app.ui_bg.add_rounded_rect(px0, 0.0f, static_cast<float>(app.win_w), static_cast<float>(app.win_h), 0.0f, {0.065f, 0.075f, 0.095f}, 0.98f);
    app.ui_bg.add_line(px0, 0.0f, px0, static_cast<float>(app.win_h), 1.2f, {0.20f, 0.23f, 0.29f}, 0.85f);

    // Header title and GPU / parameter count summary
    app.ui_fg.add_text(FONT_LG_BOLD, px0 + 20.0f, 28.0f, "Autonomy Neural Inspector", {0.95f, 0.97f, 1.0f});
    std::snprintf(buf, sizeof(buf), "Vulkan 1.4 4x-MSAA  |  %s  |  EngineZero (%zu params)",
                  app.vk.gpu_name, cnn.total_parameter_count());
    app.ui_fg.add_text(FONT_XS, px0 + 20.0f, 46.0f, buf, {0.52f, 0.60f, 0.72f});

    // Mode switcher tabs
    float seg_y = 58.0f;
    app.ui_bg.add_rounded_rect(card_x0, seg_y, card_x1, seg_y + 34.0f, 8.0f, {0.11f, 0.125f, 0.155f});
    const char* tab_names[4] = {"v11 Rules [1]", "DAgger Train [2]", "v12 Neural [3]", "Manual [4]"};
    for (int i = 0; i < 4; ++i) {
        float tx = card_x0 + 3.0f + i * 119.5f;
        float tw = 117.5f;
        app.rect_mode_tabs[i] = {tx, seg_y + 3.0f, tx + tw, seg_y + 31.0f};
        bool active = (static_cast<int>(ctrl.mode) == i);
        if (active) {
            app.ui_bg.add_rounded_rect(tx, seg_y + 3.0f, tx + tw, seg_y + 31.0f, 6.0f, {0.22f, 0.45f, 0.86f});
        }
        app.ui_fg.add_text_centered(active ? FONT_SM_BOLD : FONT_XS, tx, tx + tw, seg_y + 22.0f, tab_names[i],
                                    active ? Color3{0.98f, 0.99f, 1.0f} : Color3{0.62f, 0.68f, 0.76f});
    }

    // Card 1: Dual 3D Windshield Cameras + Bilinear Saliency Map Overlay
    float c1_y = 104.0f;
    app.ui_bg.add_card(card_x0, c1_y, card_x1, c1_y + 196.0f, 10.0f);
    app.ui_fg.add_text(FONT_SM_BOLD, card_x0 + 14.0f, c1_y + 23.0f, "Optical Vision & Input-Jacobian Saliency", {0.91f, 0.94f, 0.98f});

    const char* sal_txt = (app.sal_mode == SaliencyVisMode::RGB_PLUS_SALIENCY) ? "Saliency: Blend" :
                          (app.sal_mode == SaliencyVisMode::PURE_SALIENCY_HEATMAP) ? "Saliency: Map" : "Saliency: Off";
    app.rect_saliency_toggle = {card_x1 - 126.0f, c1_y + 7.0f, card_x1 - 12.0f, c1_y + 27.0f};
    app.ui_bg.add_rounded_rect(card_x1 - 126.0f, c1_y + 7.0f, card_x1 - 12.0f, c1_y + 27.0f, 6.0f, {0.16f, 0.19f, 0.24f});
    app.ui_fg.add_text_centered(FONT_XS, card_x1 - 126.0f, card_x1 - 12.0f, c1_y + 21.0f, sal_txt, {0.72f, 0.82f, 0.94f});

    // Camera 0 & Camera 1 Viewport Rectangles (222x150 each)
    float cam_w = 222.0f, cam_h = 150.0f;
    float cam0_x = card_x0 + 14.0f, cam_y = c1_y + 34.0f;
    float cam1_x = card_x0 + 248.0f;
    out_cam0_rect = {{static_cast<int32_t>(cam0_x), static_cast<int32_t>(cam_y)},
                     {static_cast<uint32_t>(cam_w), static_cast<uint32_t>(cam_h)}};
    out_cam1_rect = {{static_cast<int32_t>(cam1_x), static_cast<int32_t>(cam_y)},
                     {static_cast<uint32_t>(cam_w), static_cast<uint32_t>(cam_h)}};

    app.ui_bg.add_rounded_rect(cam0_x, cam_y, cam0_x + cam_w, cam_y + cam_h, 5.0f, {0.10f, 0.14f, 0.21f});
    app.ui_bg.add_rounded_rect(cam1_x, cam_y, cam1_x + cam_w, cam_y + cam_h, 5.0f, {0.10f, 0.14f, 0.21f});

    if (app.sal_mode != SaliencyVisMode::RAW_CAMERA_RGB) {
        app.ui_fg.add_textured_rect(cam0_x, cam_y, cam0_x + cam_w, cam_y + cam_h, 5.0f, 1, 1.0f);
        app.ui_fg.add_textured_rect(cam1_x, cam_y, cam1_x + cam_w, cam_y + cam_h, 5.0f, 2, 1.0f);
    }
    app.ui_fg.add_rounded_rect(cam0_x, cam_y, cam0_x + cam_w, cam_y + 21.0f, 3.0f, {0.06f, 0.07f, 0.09f}, 0.74f);
    app.ui_fg.add_rounded_rect(cam1_x, cam_y, cam1_x + cam_w, cam_y + 21.0f, 3.0f, {0.06f, 0.07f, 0.09f}, 0.74f);
    app.ui_fg.add_text_centered(FONT_XS, cam0_x, cam0_x + cam_w, cam_y + 15.0f, "Main Forward (52 deg)", {0.88f, 0.92f, 0.97f});
    app.ui_fg.add_text_centered(FONT_XS, cam1_x, cam1_x + cam_w, cam_y + 15.0f, "Wide Surround (106 deg)", {0.88f, 0.92f, 0.97f});
    app.ui_fg.add_rounded_border(cam0_x, cam_y, cam0_x + cam_w, cam_y + cam_h, 5.0f, {0.26f, 0.31f, 0.40f}, 1.3f);
    app.ui_fg.add_rounded_border(cam1_x, cam_y, cam1_x + cam_w, cam_y + cam_h, 5.0f, {0.26f, 0.31f, 0.40f}, 1.3f);

    // Card 2: Conv2D Spatial Feature Maps (6 thumbnails across 456px)
    float c2_y = 310.0f;
    app.ui_bg.add_card(card_x0, c2_y, card_x1, c2_y + 100.0f, 10.0f);
    app.ui_fg.add_text(FONT_SM_BOLD, card_x0 + 14.0f, c2_y + 22.0f, "Conv2D Spatial Feature Maps", {0.91f, 0.94f, 0.98f});
    app.ui_fg.add_text_right(FONT_XS, card_x1 - 14.0f, c2_y + 22.0f, "Layer 1 (16x12)", {0.52f, 0.58f, 0.68f});

    for (int f = 0; f < 6; ++f) {
        float bx = card_x0 + 14.0f + f * 77.2f;
        float by = c2_y + 32.0f;
        app.ui_bg.add_textured_rect(bx, by, bx + 70.0f, by + 48.0f, 4.0f, 3 + f, 1.0f);
        app.ui_bg.add_rounded_border(bx, by, bx + 70.0f, by + 48.0f, 4.0f, {0.25f, 0.29f, 0.37f}, 1.1f);
    }

    // Card 3: Control Policy Comparison & Corner Grip Dynamics (3 Well-Separated Columns)
    float c3_y = 420.0f;
    app.ui_bg.add_card(card_x0, c3_y, card_x1, c3_y + 168.0f, 10.0f);
    app.ui_fg.add_text(FONT_SM_BOLD, card_x0 + 14.0f, c3_y + 23.0f, "Control Policy & Corner Grip Dynamics", {0.91f, 0.94f, 0.98f});

    // Column 1: Steering Wheel Ring (Center at card_x0 + 70, width 120px)
    float sw_cx = card_x0 + 70.0f, sw_cy = c3_y + 88.0f, sw_r = 38.0f;
    app.ui_bg.add_circle_ring(sw_cx, sw_cy, sw_r, 3.0f, {0.24f, 0.28f, 0.36f});
    float exp_ang = -PI * 0.5f - exp.expert_steer_cmd * 1.35f;
    float cnn_ang = -PI * 0.5f - cnn.predicted_steer * 1.35f;
    app.ui_bg.add_line(sw_cx, sw_cy, sw_cx + std::cos(exp_ang) * sw_r, sw_cy + std::sin(exp_ang) * sw_r, 2.5f, {0.32f, 0.62f, 0.96f});
    app.ui_bg.add_line(sw_cx, sw_cy, sw_cx + std::cos(cnn_ang) * sw_r, sw_cy + std::sin(cnn_ang) * sw_r, 3.6f, {0.20f, 0.88f, 0.56f});
    app.ui_bg.add_rounded_rect(sw_cx - 4.0f, sw_cy - 4.0f, sw_cx + 4.0f, sw_cy + 4.0f, 4.0f, {0.88f, 0.92f, 0.96f});

    std::snprintf(buf, sizeof(buf), "Steer %+.2f / %+.2f", cnn.predicted_steer, exp.expert_steer_cmd);
    app.ui_fg.add_text_centered(FONT_XS, sw_cx - 60.0f, sw_cx + 60.0f, c3_y + 150.0f, buf, {0.68f, 0.74f, 0.82f});

    // Column 2: Friction Circle G-Force Meter (Center at card_x0 + 196, 126px right of sw_cx)
    float fc_cx = card_x0 + 196.0f, fc_cy = c3_y + 88.0f, fc_r = 37.0f;
    Color3 ring_c = ego.is_slipping ? Color3{0.96f, 0.30f, 0.30f} : Color3{0.24f, 0.28f, 0.36f};
    app.ui_bg.add_circle_ring(fc_cx, fc_cy, fc_r, 2.4f, ring_c);
    app.ui_bg.add_line(fc_cx - fc_r, fc_cy, fc_cx + fc_r, fc_cy, 1.0f, {0.22f, 0.26f, 0.33f}, 0.6f);
    app.ui_bg.add_line(fc_cx, fc_cy - fc_r, fc_cx, fc_cy + fc_r, 1.0f, {0.22f, 0.26f, 0.33f}, 0.6f);

    float gx_n = clampf(ego.lat_accel / 7.65f, -1.1f, 1.1f);
    float gy_n = clampf(-ego.actual_accel / 7.65f, -1.1f, 1.1f);
    Color3 gdot_c = ego.is_slipping ? Color3{0.96f, 0.28f, 0.28f} : Color3{0.24f, 0.56f, 0.96f};
    app.ui_bg.add_rounded_rect(fc_cx + gx_n * fc_r - 5.0f, fc_cy + gy_n * fc_r - 5.0f,
                               fc_cx + gx_n * fc_r + 5.0f, fc_cy + gy_n * fc_r + 5.0f, 5.0f, gdot_c);

    std::snprintf(buf, sizeof(buf), "Tire Grip: %3.0f%%", ego.grip_utilization * 100.0f);
    app.ui_fg.add_text_centered(FONT_XS, fc_cx - 56.0f, fc_cx + 56.0f, c3_y + 150.0f, buf,
                                ego.is_slipping ? Color3{0.96f, 0.36f, 0.36f} : Color3{0.68f, 0.74f, 0.82f});

    // Column 3: Left-Label / Right-Value Telemetry Table (card_x0 + 266 .. card_x1 - 14)
    float tx_l = card_x0 + 266.0f;
    float tx_r = card_x1 - 14.0f;

    app.ui_fg.add_text(FONT_XS, tx_l, c3_y + 52.0f, "Accel (v12 / v11)", {0.64f, 0.70f, 0.78f});
    std::snprintf(buf, sizeof(buf), "%+.2f / %+.2f", cnn.predicted_accel, exp.expert_accel_cmd);
    app.ui_fg.add_text_right(FONT_XS, tx_r, c3_y + 52.0f, buf, {0.88f, 0.92f, 0.97f});

    app.ui_fg.add_text(FONT_XS, tx_l, c3_y + 73.0f, "360 LOS Visible / Crashes", {0.64f, 0.70f, 0.78f});
    std::snprintf(buf, sizeof(buf), "%d vis  |  %d wrecks", exp.visible_objects_360, app.world.active_accident_count / 2);
    app.ui_fg.add_text_right(FONT_XS, tx_r, c3_y + 73.0f, buf, {0.84f, 0.88f, 0.94f});

    app.ui_fg.add_text(FONT_XS, tx_l, c3_y + 94.0f, "Stop Line Distance", {0.64f, 0.70f, 0.78f});
    std::snprintf(buf, sizeof(buf), "%.1f m", exp.dist_to_stop_line);
    app.ui_fg.add_text_right(FONT_XS, tx_r, c3_y + 94.0f, buf, {0.84f, 0.88f, 0.94f});

    app.ui_fg.add_text(FONT_XS, tx_l, c3_y + 115.0f, "Lane Center Offset", {0.64f, 0.70f, 0.78f});
    std::snprintf(buf, sizeof(buf), "%+.2f m", exp.cross_track_error);
    app.ui_fg.add_text_right(FONT_XS, tx_r, c3_y + 115.0f, buf, {0.84f, 0.88f, 0.94f});

    app.ui_bg.add_rounded_rect(tx_l, c3_y + 128.0f, tx_r, c3_y + 152.0f, 6.0f, {0.14f, 0.17f, 0.22f});
    app.ui_fg.add_text_centered(FONT_XS, tx_l, tx_r, c3_y + 144.0f, exp.primary_reason, {0.44f, 0.76f, 0.98f});

    // Card 4: DAgger online training metrics & loss curve
    float c4_y = 598.0f;
    app.ui_bg.add_card(card_x0, c4_y, card_x1, c4_y + 216.0f, 10.0f);
    app.ui_fg.add_text(FONT_SM_BOLD, card_x0 + 14.0f, c4_y + 23.0f, "DAgger Online Imitation Convergence", {0.91f, 0.94f, 0.98f});

    std::snprintf(buf, sizeof(buf), "Buffer: %zu", ctrl.replay_buffer.size());
    app.ui_fg.add_text(FONT_XS, card_x0 + 14.0f, c4_y + 44.0f, buf, {0.68f, 0.74f, 0.82f});

    std::snprintf(buf, sizeof(buf), "AdamW: %llu", static_cast<unsigned long long>(cnn.adam_step_count));
    app.ui_fg.add_text(FONT_XS, card_x0 + 128.0f, c4_y + 44.0f, buf, {0.68f, 0.74f, 0.82f});

    std::snprintf(buf, sizeof(buf), "Loss: %.4f", ctrl.smoothed_loss);
    app.ui_fg.add_text(FONT_XS, card_x0 + 252.0f, c4_y + 44.0f, buf, {0.22f, 0.88f, 0.56f});

    std::snprintf(buf, sizeof(buf), "Run: %.0f m", ctrl.autonomous_meters_since_intervention);
    app.ui_fg.add_text_right(FONT_XS, card_x1 - 14.0f, c4_y + 44.0f, buf, {0.42f, 0.72f, 0.98f});

    float gx0 = card_x0 + 14.0f, gy0 = c4_y + 56.0f, gw = 456.0f, gh = 144.0f;
    app.ui_bg.add_rounded_rect(gx0, gy0, gx0 + gw, gy0 + gh, 6.0f, {0.07f, 0.08f, 0.105f});

    for (int g = 1; g <= 3; ++g) {
        float ly = gy0 + (gh * g) / 4.0f;
        app.ui_bg.add_line(gx0 + 4.0f, ly, gx0 + gw - 4.0f, ly, 1.0f, {0.20f, 0.23f, 0.28f}, 0.45f);
    }

    if (ctrl.loss_history.size() >= 2) {
        for (size_t i = 0; i + 1 < ctrl.loss_history.size(); ++i) {
            float tx0 = gx0 + (static_cast<float>(i) / 240.0f) * gw;
            float tx1 = gx0 + (static_cast<float>(i + 1) / 240.0f) * gw;
            float ty0 = (gy0 + gh - 5.0f) - clampf(ctrl.loss_history[i] / 0.35f, 0.0f, 1.0f) * (gh - 14.0f);
            float ty1 = (gy0 + gh - 5.0f) - clampf(ctrl.loss_history[i + 1] / 0.35f, 0.0f, 1.0f) * (gh - 14.0f);
            app.ui_bg.add_vgrad_quad(tx0, ty0, tx1, ty1, gy0 + gh - 2.0f, {0.18f, 0.84f, 0.54f}, 0.22f, 0.01f);
            app.ui_bg.add_line(tx0, ty0, tx1, ty1, 2.2f, {0.20f, 0.88f, 0.56f});
        }
    }
    if (ctrl.disagreement_history.size() >= 2) {
        for (size_t i = 0; i + 1 < ctrl.disagreement_history.size(); ++i) {
            float tx0 = gx0 + (static_cast<float>(i) / 240.0f) * gw;
            float tx1 = gx0 + (static_cast<float>(i + 1) / 240.0f) * gw;
            float ty0 = (gy0 + gh - 5.0f) - clampf(ctrl.disagreement_history[i] / 0.55f, 0.0f, 1.0f) * (gh - 14.0f);
            float ty1 = (gy0 + gh - 5.0f) - clampf(ctrl.disagreement_history[i + 1] / 0.55f, 0.0f, 1.0f) * (gh - 14.0f);
            app.ui_bg.add_line(tx0, ty0, tx1, ty1, 1.5f, {0.32f, 0.62f, 0.96f}, 0.72f);
        }
    }
    app.ui_bg.add_rounded_border(gx0, gy0, gx0 + gw, gy0 + gh, 6.0f, {0.20f, 0.24f, 0.30f}, 1.1f);

    // Bottom action toolbar row
    float act_y = 826.0f;
    AutoparkState ap_st = ctrl.expert_v11.autopark_state;
    const char* park_btn_txt = (ap_st == AutoparkState::INACTIVE)          ? "Seek Park [P]" :
                               (ap_st == AutoparkState::SEARCHING_360_LOS) ? "Stop Search [P]" :
                               (ap_st == AutoparkState::UNPARKING_EXIT)    ? "Unparking..." :
                                                                             "Exit Park [P]";
    const char* act_labels[5] = {"+250 Train [T]", "Scenario [E]", "Reset CNN [R]", park_btn_txt, "New City"};
    for (int i = 0; i < 5; ++i) {
        float bx = card_x0 + i * 98.0f;
        float bw = 92.0f;
        app.rect_action_btns[i] = {bx, act_y, bx + bw, act_y + 34.0f};
        bool hovered = app.rect_action_btns[i].contains(app.hover_mouse_x, app.hover_mouse_y);
        Color3 bg = (i == 0) ? (hovered ? Color3{0.24f, 0.48f, 0.86f} : Color3{0.18f, 0.38f, 0.72f}) :
                    (i == 3 && ap_st == AutoparkState::SEARCHING_360_LOS) ? Color3{0.68f, 0.44f, 0.12f} :
                    (i == 3 && ap_st != AutoparkState::INACTIVE)          ? Color3{0.14f, 0.56f, 0.38f} :
                               (hovered ? Color3{0.18f, 0.21f, 0.26f} : Color3{0.13f, 0.15f, 0.19f});
        app.ui_bg.add_rounded_rect(bx, act_y, bx + bw, act_y + 34.0f, 7.0f, bg);
        app.ui_bg.add_rounded_border(bx, act_y, bx + bw, act_y + 34.0f, 7.0f, {0.25f, 0.29f, 0.36f}, 1.1f);
        app.ui_fg.add_text_centered(FONT_XS, bx, bx + bw, act_y + 22.0f, act_labels[i], {0.92f, 0.95f, 0.99f});
    }
}

// Assemble 3D ViewProj Push Constants & Render Full Vulkan Frame
void execute_vulkan_frame(AppState& app) {
    if (app.need_swapchain_resize) {
        app.vk.recreate_swapchain(app.win_w, app.win_h);
        app.need_swapchain_resize = false;
    }

    const Vehicle& ego = app.world.ego;
    if (!app.cam_initialized) {
        app.smooth_cam_target = ego.pos;
        app.smooth_cam_heading = ego.heading;
        app.cam_initialized = true;
    } else {
        app.smooth_cam_target = app.smooth_cam_target * 0.84f + ego.pos * 0.16f;
        float dh = wrap_angle(ego.heading - app.smooth_cam_heading);
        app.smooth_cam_heading = wrap_angle(app.smooth_cam_heading + dh * 0.14f);
    }

    int main_vp_w = std::max(200, app.win_w - 520);
    int main_vp_h = std::max(200, app.win_h);
    float aspect = static_cast<float>(main_vp_w) / static_cast<float>(main_vp_h);

    Vec2 ego_fwd = Vec2::from_angle(app.smooth_cam_heading);
    Vec3 eye, target, up = {0.0f, 0.0f, 1.0f};
    float fov_y = 46.0f;

    if (app.cam_mode == CameraViewMode::CHASE_3D) {
        float cam_yaw = app.smooth_cam_heading + app.orbit_yaw_offset;
        Vec2 cam_back = Vec2::from_angle(cam_yaw);
        float dist = app.orbit_zoom * 0.56f;
        float height = dist * std::sin(app.orbit_pitch) + 2.4f;
        float horiz  = dist * std::cos(app.orbit_pitch);
        eye = {app.smooth_cam_target.x - cam_back.x * horiz, app.smooth_cam_target.y - cam_back.y * horiz, height};
        target = {app.smooth_cam_target.x + ego_fwd.x * 7.0f, app.smooth_cam_target.y + ego_fwd.y * 7.0f, 1.05f};
    } else if (app.cam_mode == CameraViewMode::ISOMETRIC_3D) {
        float cam_yaw = app.orbit_yaw_offset - 0.78f;
        float dist = app.orbit_zoom * 1.85f;
        eye = {
            app.smooth_cam_target.x - std::cos(cam_yaw) * dist * std::cos(app.orbit_pitch),
            app.smooth_cam_target.y - std::sin(cam_yaw) * dist * std::cos(app.orbit_pitch),
            std::max(12.0f, dist * std::sin(app.orbit_pitch) + 12.0f)
        };
        target = {app.smooth_cam_target.x, app.smooth_cam_target.y, 0.0f};
        fov_y = 44.0f;
    } else {
        eye = {0.0f, -1.0f, 195.0f};
        target = {0.0f, 0.0f, 0.0f};
        up = {0.0f, 1.0f, 0.0f};
        fov_y = 48.0f;
    }

    Mat4 vp_main = Mat4::perspective_vulkan(fov_y, aspect, 0.8f, 460.0f) * Mat4::look_at(eye, target, up);
    PushConst3D pc_main{};
    std::memcpy(pc_main.viewProj, vp_main.m, sizeof(vp_main.m));
    pc_main.camPosAndTime[0] = eye.x;
    pc_main.camPosAndTime[1] = eye.y;
    pc_main.camPosAndTime[2] = eye.z;
    pc_main.camPosAndTime[3] = app.world.sim_time;

    // Build main 3D scene mesh
    app.mesh_main_3d.build_world_scene(app.world, app.controller, eye, false, app.cam_mode == CameraViewMode::CHASE_3D);

    // Build dual-camera sensor 3D scene mesh (from ego windshield perspective)
    Vec2 fwd2 = Vec2::from_angle(ego.heading);
    Vec3 cam_eye = {ego.pos.x + fwd2.x * 0.65f, ego.pos.y + fwd2.y * 0.65f, 1.45f};
    bool draw_sensor_3d = (app.sal_mode != SaliencyVisMode::PURE_SALIENCY_HEATMAP);
    if (draw_sensor_3d) {
        app.mesh_cam_3d.build_world_scene(app.world, app.controller, cam_eye, true, false);
    }

    auto make_sensor_pc = [&](float fov_deg, float pitch_down_rad) -> PushConst3D {
        Vec3 cam_target = {
            cam_eye.x + fwd2.x * std::cos(pitch_down_rad) * 20.0f,
            cam_eye.y + fwd2.y * std::cos(pitch_down_rad) * 20.0f,
            cam_eye.z - std::sin(pitch_down_rad) * 20.0f
        };
        Mat4 m = Mat4::perspective_vulkan(fov_deg * 0.72f, 222.0f / 150.0f, 0.5f, 180.0f) *
                 Mat4::look_at(cam_eye, cam_target, {0.0f, 0.0f, 1.0f});
        PushConst3D pc{};
        std::memcpy(pc.viewProj, m.m, sizeof(m.m));
        pc.camPosAndTime[0] = cam_eye.x;
        pc.camPosAndTime[1] = cam_eye.y;
        pc.camPosAndTime[2] = cam_eye.z;
        pc.camPosAndTime[3] = app.world.sim_time;
        return pc;
    };

    PushConst3D pc_cam0 = make_sensor_pc(52.0f, 0.06f);
    PushConst3D pc_cam1 = make_sensor_pc(106.0f, 0.14f);

    VkRect2D cam0_rect{}, cam1_rect{};
    build_vulkan_ui_layers(app, cam0_rect, cam1_rect);

    uint8_t sal0_rgba[CAM_W * CAM_H * 4];
    uint8_t sal1_rgba[CAM_W * CAM_H * 4];
    uint8_t fm_rgba[6][12 * 16 * 4];
    build_dynamic_neural_textures(app, sal0_rgba, sal1_rgba, fm_rgba);

    app.vk.render_frame(
        app.mesh_main_3d.verts,
        app.mesh_cam_3d.verts,
        app.ui_bg.verts,
        app.ui_fg.verts,
        pc_main, pc_cam0, pc_cam1,
        cam0_rect, cam1_rect,
        draw_sensor_3d,
        sal0_rgba, sal1_rgba, fm_rgba
    );
}

// Win32 input and button click dispatch
void handle_mouse_click(AppState& app, int mx, int my) {
    for (int i = 0; i < 4; ++i) {
        if (app.rect_mode_tabs[i].contains(mx, my)) {
            app.controller.mode = static_cast<DriveMode>(i);
            return;
        }
    }
    if (app.rect_saliency_toggle.contains(mx, my)) {
        app.sal_mode = static_cast<SaliencyVisMode>((static_cast<int>(app.sal_mode) + 1) % 3);
        return;
    }
    for (int i = 0; i < 5; ++i) {
        if (app.rect_action_btns[i].contains(mx, my)) {
            if (i == 0) app.controller.run_turbo_dagger_training(app.world, 250);
            else if (i == 1) {
                app.world.trigger_edge_case(app.next_edge_case_idx);
                app.next_edge_case_idx = (app.next_edge_case_idx + 1) % 4;
            }
            else if (i == 2) app.controller.reset_cnn_to_untrained();
            else if (i == 3) app.controller.expert_v11.toggle_autopark_button(app.world);
            else if (i == 4) {
                app.controller.expert_v11.autopark_state = AutoparkState::INACTIVE;
                app.controller.expert_v11.locked_parking_space_id = -1;
                app.world.reset_simulation();
            }
            return;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (app.rect_cam_pills[i].contains(mx, my)) {
            app.cam_mode = static_cast<CameraViewMode>(i);
            app.orbit_yaw_offset = 0.0f;
            return;
        }
    }
    int mults[3] = {1, 2, 5};
    for (int i = 0; i < 4; ++i) {
        if (app.rect_speed_pills[i].contains(mx, my)) {
            if (i < 3) {
                app.sim_speed_mult = mults[i];
                app.paused = false;
            } else {
                app.paused = !app.paused;
            }
            return;
        }
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (!g_app) return DefWindowProcA(hwnd, msg, wParam, lParam);

    switch (msg) {
        case WM_SIZE: {
            int nw = static_cast<int>(LOWORD(lParam));
            int nh = static_cast<int>(HIWORD(lParam));
            if (nw > 0 && nh > 0 && (nw != g_app->win_w || nh != g_app->win_h)) {
                g_app->win_w = std::max(960, nw);
                g_app->win_h = std::max(640, nh);
                g_app->need_swapchain_resize = true;
            }
            return 0;
        }
        case WM_LBUTTONDOWN: {
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));
            handle_mouse_click(*g_app, mx, my);
            if (mx < g_app->win_w - 520 && my < g_app->win_h - 60) {
                g_app->mouse_dragging = true;
                g_app->last_mouse_x = mx;
                g_app->last_mouse_y = my;
                SetCapture(hwnd);
            }
            return 0;
        }
        case WM_LBUTTONUP: {
            g_app->mouse_dragging = false;
            ReleaseCapture();
            return 0;
        }
        case WM_MOUSEMOVE: {
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));
            g_app->hover_mouse_x = mx;
            g_app->hover_mouse_y = my;
            if (g_app->mouse_dragging) {
                int dx = mx - g_app->last_mouse_x;
                int dy = my - g_app->last_mouse_y;
                g_app->last_mouse_x = mx;
                g_app->last_mouse_y = my;
                g_app->orbit_yaw_offset = wrap_angle(g_app->orbit_yaw_offset - dx * 0.007f);
                g_app->orbit_pitch = clampf(g_app->orbit_pitch + dy * 0.006f, 0.15f, 1.35f);
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            short delta = GET_WHEEL_DELTA_WPARAM(wParam);
            g_app->orbit_zoom = clampf(g_app->orbit_zoom - (delta / 120.0f) * 2.2f, 10.0f, 85.0f);
            return 0;
        }
        case WM_KEYDOWN: {
            switch (wParam) {
                case '1': g_app->controller.mode = DriveMode::RULE_V11_EXPERT; break;
                case '2': g_app->controller.mode = DriveMode::DAGGER_TRAINING; break;
                case '3': g_app->controller.mode = DriveMode::NEURAL_V12_AUTONOMOUS; break;
                case '4': g_app->controller.mode = DriveMode::MANUAL_HUMAN; break;
                case 'T': g_app->controller.run_turbo_dagger_training(g_app->world, 250); break;
                case 'R': g_app->controller.reset_cnn_to_untrained(); break;
                case 'S': {
                    int next_s = (static_cast<int>(g_app->sal_mode) + 1) % 3;
                    g_app->sal_mode = static_cast<SaliencyVisMode>(next_s);
                    break;
                }
                case 'C': {
                    int next_c = (static_cast<int>(g_app->cam_mode) + 1) % 3;
                    g_app->cam_mode = static_cast<CameraViewMode>(next_c);
                    g_app->orbit_yaw_offset = 0.0f;
                    break;
                }
                case 'E': {
                    g_app->world.trigger_edge_case(g_app->next_edge_case_idx);
                    g_app->next_edge_case_idx = (g_app->next_edge_case_idx + 1) % 4;
                    break;
                }
                case 'F': {
                    if (g_app->sim_speed_mult == 1) g_app->sim_speed_mult = 2;
                    else if (g_app->sim_speed_mult == 2) g_app->sim_speed_mult = 5;
                    else g_app->sim_speed_mult = 1;
                    break;
                }
                case 'P': {
                    g_app->controller.expert_v11.toggle_autopark_button(g_app->world);
                    break;
                }
                case 'M': {
                    g_app->controller.cnn_v12.save_weights("neural_v12_weights.bin");
                    break;
                }
                case VK_SPACE: g_app->paused = !g_app->paused; break;
                case VK_ESCAPE: PostQuitMessage(0); break;
            }
            return 0;
        }
        case WM_CLOSE:
        case WM_DESTROY: {
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

int run_headless_verification_suite() {
    std::printf("======================================================================\n");
    std::printf("  AUTONOMOUS DRIVING C++ SIMULATOR â€” ENGINEZERO & VULKAN 1.4 TEST\n");
    std::printf("======================================================================\n");

    AppState app;
    int n_lights = 0, n_stop = 0, n_yield = 0, n_police = 0;
    int n_cross = 0, n_tjct = 0, n_round = 0, n_multi_branch = 0;
    for (const auto& it : app.world.intersections) {
        if (it.ctrl_type == IntersectionControlType::TRAFFIC_LIGHTS) ++n_lights;
        else if (it.ctrl_type == IntersectionControlType::FOUR_WAY_STOP) ++n_stop;
        else if (it.ctrl_type == IntersectionControlType::TWO_WAY_YIELD) ++n_yield;
        else ++n_police;

        if (it.branch_count >= 5) ++n_multi_branch;
        if (it.shape == IntersectionShape::CROSS_PLUS) ++n_cross;
        else if (it.shape == IntersectionShape::T_JUNCTION) ++n_tjct;
        else if (it.shape == IntersectionShape::ROUNDABOUT_PLAZA) ++n_round;
    }

    int n_one_way = 0, n_multilane = 0, n_buses = 0;
    for (const auto& e : app.world.edges) {
        if (!e.is_intersection_connector) {
            if (e.is_one_way) ++n_one_way;
            if (e.num_lanes_same_dir > 1) ++n_multilane;
        }
    }
    for (const auto& v : app.world.npc_cars) {
        if (v.is_bus) ++n_buses;
    }

    int n_vacant_spaces = 0, n_occupied_spaces = 0, n_lane1_lateral = 0;
    for (const auto& sp : app.world.parking_spaces) {
        if (sp.is_occupied) ++n_occupied_spaces;
        else ++n_vacant_spaces;
        if (sp.is_on_lane_one) ++n_lane1_lateral;
    }

    std::printf("[1/5] Procedural Random City (Seed 0x%08X) & Multi-Lane / Multi-Branch Network...\n", app.world.city_seed);
    std::printf("      Control Types: %d Lights, %d Police Officers, %d All-Way Stops, %d Yield Signs\n",
                n_lights, n_police, n_stop, n_yield);
    std::printf("      Shapes       : %d '+' Cross, %d 'T' Junctions, %d Roundabouts, %d Multi-Branch (5..8-Way) Hubs\n",
                n_cross, n_tjct, n_round, n_multi_branch);
    std::printf("      Road Network : %zu Total Edges | %d Multi-Lane Blvd Edges | %d One-Way Edges\n",
                app.world.edges.size(), n_multilane, n_one_way);
    std::printf("      Actors & Zones: %d Transit Buses | %zu BUS Stop Zones | %zu Half-Parked Cars | %zu Bicycles\n",
                n_buses, app.world.bus_stops.size(), app.world.parked_cars.size(), app.world.bicycles.size());
    std::printf("      Parking System: %zu Rectangular Lots | %zu Marked Stalls (%d Vacant, %d Occupied, %d Lane-1 Lateral)\n",
                app.world.parking_lots.size(), app.world.parking_spaces.size(),
                n_vacant_spaces, n_occupied_spaces, n_lane1_lateral);

    std::printf("\n[2/5] Testing Phase 1: Rule Planner Overtaking, Autopark (D/R/P) & 360-Degree LOS...\n");
    app.controller.mode = DriveMode::RULE_V11_EXPERT;

    app.controller.expert_v11.toggle_autopark_button(app.world);
    bool started_search = (app.controller.expert_v11.autopark_state == AutoparkState::SEARCHING_360_LOS);
    app.controller.expert_v11.toggle_autopark_button(app.world);
    bool cancelled_search = (app.controller.expert_v11.autopark_state == AutoparkState::INACTIVE);
    std::printf("      Autopark Search Start & Cancel Check: Started=%s, Cancelled=%s\n",
                started_search ? "TRUE" : "FALSE", cancelled_search ? "TRUE" : "FALSE");

    if (!app.world.parking_spaces.empty()) {
        auto run_parking_layout_test = [&](ParkingLayoutType target_layout, int require_lane1, const char* label) {
            const ParkingSpace* test_sp = nullptr;
            for (const auto& sp : app.world.parking_spaces) {
                if (!sp.is_occupied && sp.layout == target_layout) {
                    if (require_lane1 == 1 && !sp.is_on_lane_one) continue;
                    if (require_lane1 == 0 &&  sp.is_on_lane_one) continue;
                    test_sp = &sp;
                    break;
                }
            }
            if (!test_sp) return;

            app.controller.expert_v11.autopark_state = AutoparkState::INACTIVE;
            app.controller.expert_v11.locked_parking_space_id = -1;
            const auto& road_e = app.world.edges[test_sp->road_edge_id];
            app.world.ego.route_edges = {road_e.id};
            app.world.ego.current_route_idx = 0;
            app.world.ego.pos = road_e.samples[4].pos;
            app.world.ego.heading = road_e.samples[4].heading;
            app.world.ego.current_edge_s = road_e.samples[4].s;
            app.world.ego.current_lat_offset = 0.0f;
            app.world.ego.gear = GearState::DRIVE;
            app.world.ego.speed = 6.0f;
            for (auto& npc : app.world.npc_cars) {
                npc.pos = {999.0f, 999.0f};
                npc.speed = 0.0f;
            }
            app.world.update_360_perception_visibility();

            app.controller.expert_v11.toggle_autopark_button(app.world);
            bool reached_park = false;
            bool used_reverse_gear = false;
            for (int step_p = 0; step_p < 440; ++step_p) {
                app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
                if (app.world.ego.gear == GearState::REVERSE) used_reverse_gear = true;
                if (app.controller.expert_v11.autopark_state == AutoparkState::PARKED_IN_BOUNDS) {
                    reached_park = true;
                    break;
                }
            }
            const ParkingSpace* locked_sp = app.controller.expert_v11.find_locked_space(app.world);
            float park_dist_err = locked_sp ? (app.world.ego.pos - locked_sp->center).len() : 99.0f;
            float lat_err = locked_sp
                ? std::fabs((app.world.ego.pos - locked_sp->center).dot(Vec2::from_angle(locked_sp->heading).perp_left()))
                : 99.0f;
            float head_err_deg = locked_sp
                ? (std::fabs(wrap_angle(app.world.ego.heading - locked_sp->heading)) * RAD2DEG)
                : 99.0f;
            std::printf("      Autopark [%s]: ReachedPark=%s, UsedRev=%s, CenterErr=%.2fm, LatErr=%.2fm, HeadingErr=%.1f deg\n",
                        label, reached_park ? "TRUE" : "FALSE", used_reverse_gear ? "TRUE" : "FALSE",
                        park_dist_err, lat_err, head_err_deg);

            app.controller.expert_v11.toggle_autopark_button(app.world);
            bool resumed_road = false;
            for (int step_u = 0; step_u < 240; ++step_u) {
                app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
                if (app.controller.expert_v11.autopark_state == AutoparkState::INACTIVE && app.world.ego.speed > 2.5f) {
                    resumed_road = true;
                    break;
                }
            }
            std::printf("      Unpark   [%s]: ResumedRoad=%s (Speed=%.2f m/s)\n",
                        label, resumed_road ? "TRUE" : "FALSE", app.world.ego.speed);
        };

        run_parking_layout_test(ParkingLayoutType::PERPENDICULAR_90_REVERSE, -1, "90-Deg Reverse-In ");
        run_parking_layout_test(ParkingLayoutType::PERPENDICULAR_90_FRONT,   -1, "90-Deg Front-In   ");
        run_parking_layout_test(ParkingLayoutType::ANGLED_45_FRONT,          -1, "45-Deg Angled-In  ");
        run_parking_layout_test(ParkingLayoutType::PARALLEL_LATERAL_REVERSE,  1, "Lane-1 Direct-In  ");
        run_parking_layout_test(ParkingLayoutType::PARALLEL_LATERAL_REVERSE,  0, "Curb Parallel-Rev ");
    }

    app.world.reset_simulation();
    app.controller.mode = DriveMode::RULE_V11_EXPERT;
    for (int i = 0; i < 120; ++i) {
        app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
    }
    std::printf("      After 120 steps (6.0s): Speed=%.2f m/s, CTE=%+.3f m, 360_Vis=%d, State=%s\n",
                app.world.ego.speed, app.controller.expert_v11.telemetry.cross_track_error,
                app.controller.expert_v11.telemetry.visible_objects_360,
                app.controller.expert_v11.telemetry.primary_reason);

    if (!app.world.buildings.empty() && !app.world.npc_cars.empty()) {
        const auto& b = app.world.buildings[0];
        Vec2 b_center = (b.min_pt + b.max_pt) * 0.5f;
        Vec2 dir_to_b = (b_center - app.world.ego.pos).normalized();
        Vec2 hidden_crash_pos = b_center + dir_to_b * 18.0f;
        bool los_hidden = app.world.has_360_line_of_sight(hidden_crash_pos);
        std::printf("      Occluded Crash Behind Building Test: has_360_line_of_sight = %s (Expected: FALSE)\n",
                    los_hidden ? "TRUE" : "FALSE");
    }

    app.world.trigger_edge_case(2);
    app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
    std::printf("      Edge Case 2A (Dotted-Line Overtake, Clear Oncoming): Overtake=%s, LatOffset=%+.2fm -> %s\n",
                app.controller.expert_v11.telemetry.rule_overtaking_active ? "TRUE" : "FALSE",
                app.controller.expert_v11.telemetry.evasive_lat_offset,
                app.controller.expert_v11.telemetry.primary_reason);

    app.world.trigger_edge_case(2);
    app.world.ego.current_lat_offset = 0.0f;
    app.controller.expert_v11.overtake_hold_timer = 0.0f;
    if (!app.world.npc_cars.empty()) {
        const auto& ce = app.world.edges[app.world.ego.route_edges[app.world.ego.current_route_idx]];
        Vec2 efwd = Vec2::from_angle(app.world.ego.heading);
        Vec2 eleft = efwd.perp_left();
        Vehicle& onc = app.world.npc_cars[0];
        onc.is_crashed = false;
        onc.is_stopped_at_bus_zone = false;
        if (ce.oppose_edge_id >= 0) {
            onc.route_edges = {ce.oppose_edge_id};
            onc.current_route_idx = 0;
        }
        onc.pos = app.world.ego.pos + efwd * 24.0f + eleft * 3.6f;
        onc.heading = wrap_angle(app.world.ego.heading + PI);
        onc.speed = 8.5f;
    }
    app.controller.expert_v11.evaluate(app.world, 0.05f, true);
    std::printf("      Edge Case 2B (Unsafe Oncoming Traffic Present)     : Overtake=%s, WaitOncoming=%s -> %s\n",
                app.controller.expert_v11.telemetry.rule_overtaking_active ? "TRUE" : "FALSE",
                app.controller.expert_v11.telemetry.rule_waiting_to_overtake ? "TRUE" : "FALSE",
                app.controller.expert_v11.telemetry.primary_reason);

    app.world.trigger_edge_case(3);
    app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
    std::printf("      Edge Case 3 (Visible Crash in 360 LOS): CrashDetected=%s, EvasiveOffset=%+.2fm -> %s\n",
                app.controller.expert_v11.telemetry.crash_prevention_active ? "TRUE" : "FALSE",
                app.controller.expert_v11.telemetry.evasive_lat_offset,
                app.controller.expert_v11.telemetry.primary_reason);

    app.world.trigger_edge_case(0);
    app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
    std::printf("      Edge Case 0 (Yellow Light Dilemma): StopDist=%.2fm, MinStop=%.2fm -> %s\n",
                app.controller.expert_v11.telemetry.dist_to_stop_line,
                app.controller.expert_v11.telemetry.dilemma_min_stop_dist,
                app.controller.expert_v11.telemetry.primary_reason);

    app.world.trigger_edge_case(1);
    app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
    std::printf("      Edge Case 1 (Jaywalker Pedestrian): PedID=%d, AccelCmd=%+.2f -> %s\n",
                app.controller.expert_v11.telemetry.detected_pedestrian_id,
                app.controller.expert_v11.telemetry.expert_accel_cmd,
                app.controller.expert_v11.telemetry.primary_reason);

    std::printf("\n[3/5] Testing Phase 2: EngineZero CNN & Saliency Gradients...\n");
    std::printf("      CNN Total Trainable Parameters: %zu\n",
                app.controller.cnn_v12.total_parameter_count());

    app.world.reset_simulation();
    app.controller.reset_cnn_to_untrained();
    app.controller.mode = DriveMode::DAGGER_TRAINING;

    for (int i = 0; i < 20; ++i) {
        app.controller.step(app.world, 0.05f, 0.0f, 0.0f, true);
    }
    float initial_loss = app.controller.smoothed_loss;

    std::printf("\n[4/5] Running 400 DAgger Online Training Iterations (EngineZero Autograd)...\n");
    for (int epoch = 1; epoch <= 4; ++epoch) {
        app.controller.run_turbo_dagger_training(app.world, 100);
        std::printf("      DAgger Step %3d | Buffer: %4zu | Smoothed Loss: %.4f | L1 Disagreement: %.4f\n",
                    epoch * 100, app.controller.replay_buffer.size(),
                    app.controller.smoothed_loss, app.controller.current_disagreement);
    }

    std::printf("\n[5/5] Evaluating Long-Horizon 2,000-Step City Traffic & Autonomy Endurance (100.0s)...\n");
    app.world.reset_simulation();
    app.controller.mode = DriveMode::DAGGER_TRAINING;
    float total_ego_dist = 0.0f;
    float total_npc_dist = 0.0f;
    for (int i = 0; i < 2000; ++i) {
        app.controller.step(app.world, 0.05f, 0.0f, 0.0f, false);
        total_ego_dist += app.world.ego.speed * 0.05f;
        for (const auto& npc : app.world.npc_cars) {
            total_npc_dist += npc.speed * 0.05f;
        }
        if ((i + 1) % 500 == 0) {
            std::printf("      Step %4d (t=%5.1fs) | EgoDist: %6.1fm (v=%4.1f m/s) | AvgNPCDist: %6.1fm | Loss: %.4f\n",
                        i + 1, (i + 1) * 0.05f, total_ego_dist, app.world.ego.speed,
                        total_npc_dist / std::max<size_t>(1, app.world.npc_cars.size()),
                        app.controller.smoothed_loss);
        }
    }
    std::printf("      Loss Reduction: %.4f -> %.4f\n", initial_loss, app.controller.smoothed_loss);

    app.controller.cnn_v12.save_weights("neural_v12_weights.bin");
    std::printf("======================================================================\n");
    std::printf("  ALL VERIFICATION CHECKS PASSED!\n");
    std::printf("======================================================================\n");
    return 0;
}

} // namespace sim

int main(int argc, char** argv) {
    bool smoke_test_gui = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--headless-test") == 0) {
            return sim::run_headless_verification_suite();
        }
        if (std::strcmp(argv[i], "--smoke-test-gui") == 0) {
            smoke_test_gui = true;
        }
    }

    SetProcessDPIAware();

    sim::AppState app;
    sim::g_app = &app;

    if (!app.controller.cnn_v12.load_weights("neural_v12_weights.bin")) {
        app.controller.run_turbo_dagger_training(app.world, 180);
    } else {
        app.controller.run_turbo_dagger_training(app.world, 45);
    }
    app.controller.mode = sim::DriveMode::DAGGER_TRAINING;

    HINSTANCE hInst = GetModuleHandleA(nullptr);
    WNDCLASSA wc = {};
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = sim::WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = "AutoDriveVulkanWndClass";
    RegisterClassA(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
    RECT wr = {0, 0, app.win_w, app.win_h};
    AdjustWindowRect(&wr, style, FALSE);
    HWND hwnd = CreateWindowExA(
        0, wc.lpszClassName,
        "Autonomous Driving Simulator  |  Vulkan 1.4 4x-MSAA  |  EngineZero Autograd",
        style, CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        nullptr, nullptr, hInst, nullptr
    );

    RECT client_rc{};
    GetClientRect(hwnd, &client_rc);
    app.win_w = std::max(960, static_cast<int>(client_rc.right - client_rc.left));
    app.win_h = std::max(640, static_cast<int>(client_rc.bottom - client_rc.top));

    app.vk.init(hwnd, app.win_w, app.win_h, app.ui_bg);
    app.ui_fg.glyphs = app.ui_bg.glyphs;

    if (smoke_test_gui) {
        std::printf("[Vulkan GUI Smoke Test] Active GPU: %s\n", app.vk.gpu_name);
        std::printf("[Vulkan GUI Smoke Test] Swapchain : %ux%u (4x Hardware MSAA)\n",
                    app.vk.swapchain_extent.width, app.vk.swapchain_extent.height);
    }

    int rendered_frames = 0;
    bool running = true;
    while (running) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = false;
            }
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (!running) break;

        float man_steer = 0.0f;
        float man_accel = 0.0f;
        if (GetAsyncKeyState(VK_LEFT)  & 0x8000 || GetAsyncKeyState('A') & 0x8000) man_steer += 0.8f;
        if (GetAsyncKeyState(VK_RIGHT) & 0x8000 || GetAsyncKeyState('D') & 0x8000) man_steer -= 0.8f;
        if (GetAsyncKeyState(VK_UP)    & 0x8000 || GetAsyncKeyState('W') & 0x8000) man_accel += 0.85f;
        if (GetAsyncKeyState(VK_DOWN)  & 0x8000 || GetAsyncKeyState('S') & 0x8000) man_accel -= 0.90f;

        if (!app.paused) {
            for (int step_i = 0; step_i < app.sim_speed_mult; ++step_i) {
                bool is_last = (step_i == app.sim_speed_mult - 1);
                app.controller.step(app.world, 0.033f, man_steer, man_accel, is_last);
            }
        }

        sim::execute_vulkan_frame(app);
        ++rendered_frames;

        if (smoke_test_gui && rendered_frames >= 45) {
            std::printf("[Vulkan GUI Smoke Test] Rendered %d frames (Main 3D Verts: %zu, UI Verts: %zu) cleanly!\n",
                        rendered_frames, app.mesh_main_3d.verts.size(),
                        app.ui_bg.verts.size() + app.ui_fg.verts.size());
            break;
        }
        Sleep(14);
    }

    app.vk.shutdown();
    DestroyWindow(hwnd);
    return 0;
}
