#pragma once
// Procedural 3D mesh builder for the city world, vehicles, road markings,
// traffic signals, parking zones, and perception bounding boxes.
#include "dagger_trainer.hpp"
#include <vector>
#include <cmath>

namespace sim {

struct Vertex3D {
    float px, py, pz;
    float nx, ny, nz;
    float r, g, b, a;
    float u, v;
    float mat_type;
    // 0 = Matte lit
    // 1 = Automotive clearcoat / glass
    // 2 = Unlit / emissive (signals, LEDs, markings, trajectory overlays)
    // 3 = Soft ground shadow
    // 4 = Procedural asphalt
    // 5 = Curtain-wall building facade
    // 6 = Sky dome
};

struct Mat4 {
    float m[16] = {};

    static Mat4 perspective_vulkan(float fov_y_deg, float aspect, float z_near, float z_far) {
        Mat4 res{};
        float f = 1.0f / std::tan(fov_y_deg * DEG2RAD * 0.5f);
        res.m[0]  = f / std::max(0.1f, aspect);
        res.m[5]  = -f;
        res.m[10] = z_far / (z_near - z_far);
        res.m[11] = -1.0f;
        res.m[14] = (z_near * z_far) / (z_near - z_far);
        return res;
    }

    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 fwd = (target - eye).normalized();
        Vec3 s = fwd.cross(up).normalized();
        Vec3 u = s.cross(fwd);

        Mat4 res{};
        res.m[0] = s.x;    res.m[4] = s.y;    res.m[8]  = s.z;    res.m[12] = -s.dot(eye);
        res.m[1] = u.x;    res.m[5] = u.y;    res.m[9]  = u.z;    res.m[13] = -u.dot(eye);
        res.m[2] = -fwd.x; res.m[6] = -fwd.y; res.m[10] = -fwd.z; res.m[14] = fwd.dot(eye);
        res.m[3] = 0.0f;   res.m[7] = 0.0f;   res.m[11] = 0.0f;   res.m[15] = 1.0f;
        return res;
    }

    Mat4 operator*(const Mat4& rhs) const {
        Mat4 out{};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                out.m[col * 4 + row] =
                    m[0 * 4 + row] * rhs.m[col * 4 + 0] +
                    m[1 * 4 + row] * rhs.m[col * 4 + 1] +
                    m[2 * 4 + row] * rhs.m[col * 4 + 2] +
                    m[3 * 4 + row] * rhs.m[col * 4 + 3];
            }
        }
        return out;
    }
};

class MeshBuilder3D {
public:
    std::vector<Vertex3D> verts;

    void clear() { verts.clear(); }

    void push_tri(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 n0, Vec3 n1, Vec3 n2,
                  Color3 c, float a, float mat, Vec2 uv0 = {}, Vec2 uv1 = {}, Vec2 uv2 = {}) {
        verts.push_back({p0.x, p0.y, p0.z, n0.x, n0.y, n0.z, c.r, c.g, c.b, a, uv0.x, uv0.y, mat});
        verts.push_back({p1.x, p1.y, p1.z, n1.x, n1.y, n1.z, c.r, c.g, c.b, a, uv1.x, uv1.y, mat});
        verts.push_back({p2.x, p2.y, p2.z, n2.x, n2.y, n2.z, c.r, c.g, c.b, a, uv2.x, uv2.y, mat});
    }

    void push_quad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 n, Color3 c, float a = 1.0f, float mat = 0.0f) {
        push_tri(p0, p1, p2, n, n, n, c, a, mat, {-1,-1}, {1,-1}, {1,1});
        push_tri(p0, p2, p3, n, n, n, c, a, mat, {-1,-1}, {1,1}, {-1,1});
    }

    void push_smooth_quad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                          Vec3 n0, Vec3 n1, Vec3 n2, Vec3 n3,
                          Color3 c, float a = 1.0f, float mat = 1.0f) {
        push_tri(p0, p1, p2, n0, n1, n2, c, a, mat);
        push_tri(p0, p2, p3, n0, n2, n3, c, a, mat);
    }

    void add_box(Vec3 min_p, Vec3 max_p, Color3 col, float alpha = 1.0f, float mat = 0.0f) {
        push_quad({min_p.x, min_p.y, max_p.z}, {max_p.x, min_p.y, max_p.z},
                  {max_p.x, max_p.y, max_p.z}, {min_p.x, max_p.y, max_p.z}, {0, 0, 1}, col, alpha, mat);
        push_quad({min_p.x, min_p.y, min_p.z}, {max_p.x, min_p.y, min_p.z},
                  {max_p.x, min_p.y, max_p.z}, {min_p.x, min_p.y, max_p.z}, {0, -1, 0}, col, alpha, mat);
        push_quad({max_p.x, max_p.y, min_p.z}, {min_p.x, max_p.y, min_p.z},
                  {min_p.x, max_p.y, max_p.z}, {max_p.x, max_p.y, max_p.z}, {0, 1, 0}, col, alpha, mat);
        push_quad({max_p.x, min_p.y, min_p.z}, {max_p.x, max_p.y, min_p.z},
                  {max_p.x, max_p.y, max_p.z}, {max_p.x, min_p.y, max_p.z}, {1, 0, 0}, col, alpha, mat);
        push_quad({min_p.x, max_p.y, min_p.z}, {min_p.x, min_p.y, min_p.z},
                  {min_p.x, min_p.y, max_p.z}, {min_p.x, max_p.y, max_p.z}, {-1, 0, 0}, col, alpha, mat);
    }

    void add_tapered_oriented_box(Vec2 center_bot, Vec2 center_top, float heading,
                                  float len_bot, float wid_bot, float len_top, float wid_top,
                                  float z0, float z1, Color3 col, float alpha = 1.0f, float mat = 0.0f) {
        Vec2 dir = Vec2::from_angle(heading);
        Vec2 left = dir.perp_left();

        Vec2 b0 = center_bot - dir * (len_bot * 0.5f) - left * (wid_bot * 0.5f);
        Vec2 b1 = center_bot + dir * (len_bot * 0.5f) - left * (wid_bot * 0.5f);
        Vec2 b2 = center_bot + dir * (len_bot * 0.5f) + left * (wid_bot * 0.5f);
        Vec2 b3 = center_bot - dir * (len_bot * 0.5f) + left * (wid_bot * 0.5f);

        Vec2 t0 = center_top - dir * (len_top * 0.5f) - left * (wid_top * 0.5f);
        Vec2 t1 = center_top + dir * (len_top * 0.5f) - left * (wid_top * 0.5f);
        Vec2 t2 = center_top + dir * (len_top * 0.5f) + left * (wid_top * 0.5f);
        Vec2 t3 = center_top - dir * (len_top * 0.5f) + left * (wid_top * 0.5f);

        push_quad({t0.x, t0.y, z1}, {t1.x, t1.y, z1}, {t2.x, t2.y, z1}, {t3.x, t3.y, z1}, {0, 0, 1}, col, alpha, mat);
        push_quad({b1.x, b1.y, z0}, {b2.x, b2.y, z0}, {t2.x, t2.y, z1}, {t1.x, t1.y, z1}, Vec3{dir.x, dir.y, 0.25f}.normalized(), col, alpha, mat);
        push_quad({b3.x, b3.y, z0}, {b0.x, b0.y, z0}, {t0.x, t0.y, z1}, {t3.x, t3.y, z1}, Vec3{-dir.x, -dir.y, 0.25f}.normalized(), col, alpha, mat);
        push_quad({b0.x, b0.y, z0}, {b1.x, b1.y, z0}, {t1.x, t1.y, z1}, {t0.x, t0.y, z1}, Vec3{-left.x, -left.y, 0.2f}.normalized(), col, alpha, mat);
        push_quad({b2.x, b2.y, z0}, {b3.x, b3.y, z0}, {t3.x, t3.y, z1}, {t2.x, t2.y, z1}, Vec3{left.x, left.y, 0.2f}.normalized(), col, alpha, mat);
    }

    void add_oriented_box(Vec2 center, float heading, float len, float wid, float z0, float z1,
                          Color3 col, float alpha = 1.0f, float mat = 0.0f) {
        add_tapered_oriented_box(center, center, heading, len, wid, len, wid, z0, z1, col, alpha, mat);
    }

    void add_3d_wire_box(Vec2 center, float heading, float len, float wid, float z0, float z1,
                         float thickness, Color3 col, float alpha = 0.85f) {
        Vec2 fwd = Vec2::from_angle(heading);
        Vec2 left = fwd.perp_left();
        float hl = len * 0.5f;
        float hw = wid * 0.5f;

        Vec2 corners[4] = {
            center + fwd * hl + left * hw,
            center + fwd * hl - left * hw,
            center - fwd * hl - left * hw,
            center - fwd * hl + left * hw
        };
        for (int i = 0; i < 4; ++i) {
            add_oriented_box(corners[i], heading, thickness, thickness, z0, z1, col, alpha, 2.0f);
        }
        for (int z_pass = 0; z_pass < 2; ++z_pass) {
            float za = (z_pass == 0) ? z0 : (z1 - thickness);
            float zb = za + thickness;
            add_oriented_box(center + fwd * hl, heading, thickness, wid, za, zb, col, alpha, 2.0f);
            add_oriented_box(center - fwd * hl, heading, thickness, wid, za, zb, col, alpha, 2.0f);
            add_oriented_box(center + left * hw, heading, len, thickness, za, zb, col, alpha, 2.0f);
            add_oriented_box(center - left * hw, heading, len, thickness, za, zb, col, alpha, 2.0f);
        }
    }

    void add_smooth_sphere(Vec3 center, float rx, float ry, float rz, Color3 col,
                           int rings = 7, int sectors = 12, float mat = 0.0f, float alpha = 1.0f) {
        for (int r = 0; r < rings; ++r) {
            float phi0 = PI * (-0.5f + static_cast<float>(r) / rings);
            float phi1 = PI * (-0.5f + static_cast<float>(r + 1) / rings);
            float z0 = std::sin(phi0), zr0 = std::cos(phi0);
            float z1 = std::sin(phi1), zr1 = std::cos(phi1);

            for (int s = 0; s < sectors; ++s) {
                float th0 = TWO_PI * static_cast<float>(s) / sectors;
                float th1 = TWO_PI * static_cast<float>(s + 1) / sectors;

                Vec3 n00 = {std::cos(th0) * zr0, std::sin(th0) * zr0, z0};
                Vec3 n10 = {std::cos(th1) * zr0, std::sin(th1) * zr0, z0};
                Vec3 n11 = {std::cos(th1) * zr1, std::sin(th1) * zr1, z1};
                Vec3 n01 = {std::cos(th0) * zr1, std::sin(th0) * zr1, z1};

                Vec3 p00 = {center.x + n00.x * rx, center.y + n00.y * ry, center.z + n00.z * rz};
                Vec3 p10 = {center.x + n10.x * rx, center.y + n10.y * ry, center.z + n10.z * rz};
                Vec3 p11 = {center.x + n11.x * rx, center.y + n11.y * ry, center.z + n11.z * rz};
                Vec3 p01 = {center.x + n01.x * rx, center.y + n01.y * ry, center.z + n01.z * rz};

                push_smooth_quad(p00, p10, p11, p01, n00, n10, n11, n01, col, alpha, mat);
            }
        }
    }

    // 3D Painted Directional Pavement Arrow (for One-Way Streets & Multi-Lane Boulevards)
    void add_pavement_arrow_3d(Vec2 center, float heading, Color3 col = {0.94f, 0.96f, 1.00f}) {
        Vec2 fwd = Vec2::from_angle(heading);
        Vec2 left = fwd.perp_left();
        // Arrow shaft + 3D chevron wings
        add_oriented_box(center - fwd * 0.35f, heading, 1.50f, 0.24f, 0.01f, 0.055f, col, 0.96f, 2.0f);
        add_oriented_box(center + fwd * 0.48f + left * 0.22f, heading - 0.58f, 0.78f, 0.22f, 0.01f, 0.056f, col, 0.96f, 2.0f);
        add_oriented_box(center + fwd * 0.48f - left * 0.22f, heading + 0.58f, 0.78f, 0.22f, 0.01f, 0.056f, col, 0.96f, 2.0f);
    }

    // Designated "BUS" Pavement Stop Zone (Strictly inside Lane 1 â€” Outermost Curb Lane, NEVER on Lane 2)
    void add_bus_stop_zone_3d(const BusStopZone& bsz) {
        Vec2 fwd = Vec2::from_angle(bsz.heading);
        Vec2 left = fwd.perp_left();
        float hl = bsz.length * 0.5f; // 6.75m
        float hw = 1.02f;             // 2.04m total width -> fits 100% inside Lane 1 without touching Lane 2
        Color3 bus_yellow = {0.98f, 0.78f, 0.10f};

        // 1. Yellow Dashed/Solid Bus Bay Box Outline on the Asphalt (z = +0.056m)
        add_oriented_box(bsz.pos + fwd * hl, bsz.heading, 0.20f, hw * 2.0f, 0.01f, 0.056f, bus_yellow, 0.95f, 2.0f);
        add_oriented_box(bsz.pos - fwd * hl, bsz.heading, 0.20f, hw * 2.0f, 0.01f, 0.056f, bus_yellow, 0.95f, 2.0f);
        for (int d = -3; d <= 3; ++d) {
            add_oriented_box(bsz.pos + fwd * (d * 1.85f) + left * hw, bsz.heading, 1.15f, 0.18f, 0.01f, 0.056f, bus_yellow, 0.95f, 2.0f);
        }

        // 2. Painted 3D Block Letters "B U S" on the Pavement Surface (strictly inside Lane 1)
        auto stroke_box = [&](Vec2 c, float l_fwd, float w_lat) {
            add_oriented_box(c, bsz.heading, l_fwd, w_lat, 0.01f, 0.058f, bus_yellow, 0.98f, 2.0f);
        };

        // Letter 'B' (centered at +left * 0.56m)
        Vec2 cB = bsz.pos + left * 0.56f;
        stroke_box(cB + left * 0.16f, 1.38f, 0.10f); // left spine
        stroke_box(cB - left * 0.16f, 1.22f, 0.10f); // right loops
        stroke_box(cB + fwd * 0.64f,  0.13f, 0.38f); // top bar
        stroke_box(cB,                0.13f, 0.38f); // middle bar
        stroke_box(cB - fwd * 0.64f,  0.13f, 0.38f); // bottom bar

        // Letter 'U' (centered at bsz.pos)
        Vec2 cU = bsz.pos;
        stroke_box(cU + left * 0.16f, 1.38f, 0.10f); // left vertical
        stroke_box(cU - left * 0.16f, 1.38f, 0.10f); // right vertical
        stroke_box(cU - fwd * 0.64f,  0.13f, 0.40f); // bottom bar

        // Letter 'S' (centered at -left * 0.56m)
        Vec2 cS = bsz.pos - left * 0.56f;
        stroke_box(cS + fwd * 0.64f,  0.13f, 0.38f); // top bar
        stroke_box(cS,                0.13f, 0.38f); // middle bar
        stroke_box(cS - fwd * 0.64f,  0.13f, 0.38f); // bottom bar
        stroke_box(cS + fwd * 0.32f + left * 0.15f, 0.66f, 0.10f); // upper-left vertical
        stroke_box(cS - fwd * 0.32f - left * 0.15f, 0.66f, 0.10f); // lower-right vertical

        // 3. 3D Glass-Roofed Transit Bus Shelter on the Sidewalk beside Lane 1
        Vec2 sh_center = bsz.pos - left * 2.45f;
        // 4 dark-steel shelter support pillars
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                Vec2 post = sh_center + fwd * (sx * 2.1f) + left * (sy * 0.65f);
                add_oriented_box(post, bsz.heading, 0.10f, 0.10f, 0.06f, 2.55f, {0.20f, 0.23f, 0.28f}, 1.0f, 1.0f);
            }
        }
        // Translucent glass back wall & canopy roof
        add_oriented_box(sh_center - left * 0.65f, bsz.heading, 4.30f, 0.06f, 0.15f, 2.50f, {0.22f, 0.38f, 0.55f}, 0.55f, 1.0f);
        add_oriented_box(sh_center, bsz.heading, 4.65f, 1.55f, 2.52f, 2.66f, {0.18f, 0.26f, 0.36f}, 0.88f, 1.0f);
        // Passenger waiting bench inside shelter
        add_oriented_box(sh_center - left * 0.35f, bsz.heading, 2.80f, 0.45f, 0.46f, 0.56f, {0.58f, 0.40f, 0.22f}, 1.0f, 0.0f);
        // Blue & Yellow "BUS STOP" Totem Signpost at front of shelter
        Vec2 totem = sh_center + fwd * 2.65f + left * 0.55f;
        add_oriented_box(totem, bsz.heading, 0.08f, 0.08f, 0.06f, 2.65f, {0.30f, 0.34f, 0.40f}, 1.0f, 1.0f);
        add_oriented_box(totem, bsz.heading, 0.12f, 0.42f, 2.10f, 2.65f, {0.12f, 0.48f, 0.92f}, 1.0f, 2.0f);
        add_oriented_box(totem, bsz.heading, 0.14f, 0.36f, 2.45f, 2.60f, bus_yellow, 1.0f, 2.0f);
    }

    // 3D Rectangular Off-Street Parking Lot Zone (Paved Asphalt Rectangle + Perimeter Curbs + Driveway Apron + "P" Sign)
    void add_parking_lot_zone_3d(const ParkingLotZone& lot) {
        Vec2 fwd = Vec2::from_angle(lot.heading);
        Vec2 right = fwd.perp_right();
        float hl = lot.length * 0.5f; // 11.5m
        float hw = lot.width * 0.5f;  // 6.1m

        // 1. Concrete foundation apron + Dark procedural asphalt pad for the Rectangular Parking Lot (top at z = 0.00m)
        add_oriented_box(lot.center, lot.heading, lot.length + 1.8f, lot.width + 1.8f, -0.07f, -0.008f, {0.34f, 0.36f, 0.40f}, 1.0f, 0.0f);
        add_oriented_box(lot.center - right * 1.1f, lot.heading, lot.length + 0.6f, lot.width + 2.4f, -0.04f, 0.00f, {0.16f, 0.175f, 0.205f}, 1.0f, 4.0f);

        // 2. Raised Concrete Perimeter Curbs along the back wall and two end walls of the Rectangular Lot
        Color3 curb_col = {0.42f, 0.45f, 0.50f};
        add_oriented_box(lot.center + right * (hw + 0.35f), lot.heading, lot.length + 1.0f, 0.65f, -0.04f, 0.12f, curb_col, 1.0f, 0.0f);
        add_oriented_box(lot.center + fwd * (hl + 0.35f),   lot.heading, 0.65f, lot.width + 0.6f,  -0.04f, 0.12f, curb_col, 1.0f, 0.0f);
        add_oriented_box(lot.center - fwd * (hl + 0.35f),   lot.heading, 0.65f, lot.width + 0.6f,  -0.04f, 0.12f, curb_col, 1.0f, 0.0f);

        // 3. White Rectangular Perimeter Frame & Dashed Aisle Guidance Line inside the Lot
        Color3 line_w = {0.88f, 0.92f, 0.98f};
        add_oriented_box(lot.center + right * (hw - 0.12f), lot.heading, lot.length, 0.15f, 0.005f, 0.053f, line_w, 0.92f, 2.0f);
        Vec2 aisle_line = lot.center - right * 3.15f;
        for (float u = -9.0f; u <= 9.0f; u += 3.6f) {
            add_oriented_box(aisle_line + fwd * u, lot.heading, 1.8f, 0.13f, 0.005f, 0.053f, {0.72f, 0.78f, 0.88f}, 0.85f, 2.0f);
        }

        // 4. 3D Blue-and-White "P" Parking Zone Totem Signpost at the Lot Entrance
        Vec2 sign_pos = lot.center - fwd * (hl + 0.2f) - right * (hw - 0.4f);
        add_box({sign_pos.x - 0.07f, sign_pos.y - 0.07f, 0.0f},
                {sign_pos.x + 0.07f, sign_pos.y + 0.07f, 2.75f}, {0.30f, 0.34f, 0.40f}, 1.0f, 1.0f);
        add_oriented_box(sign_pos, lot.heading, 0.14f, 0.62f, 2.05f, 2.72f, {0.10f, 0.46f, 0.94f}, 1.0f, 2.0f);
        // White 'P' emblem on the blue sign
        add_oriented_box(sign_pos + right * 0.12f, lot.heading, 0.16f, 0.10f, 2.16f, 2.62f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
        add_oriented_box(sign_pos - right * 0.02f, lot.heading, 0.16f, 0.28f, 2.52f, 2.62f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
        add_oriented_box(sign_pos - right * 0.02f, lot.heading, 0.16f, 0.28f, 2.34f, 2.44f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
        add_oriented_box(sign_pos - right * 0.12f, lot.heading, 0.16f, 0.10f, 2.34f, 2.62f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
    }

    // 3D Marked Rectangular Parking Stall (Lateral Parallel, 90-deg Reverse-In, 90-deg Front-In, or 45-deg Angled)
    void add_parking_space_3d(const ParkingSpace& sp, const RuleTelemetry& tel, bool is_sensor_camera) {
        Vec2 fwd = Vec2::from_angle(sp.heading);
        Vec2 left = fwd.perp_left();
        float hl = sp.length * 0.5f;
        float hw = sp.width * 0.5f;

        // If this is a recessed curbside lateral stall (lot_id == -1 && !is_on_lane_one), draw its paved asphalt pocket
        if (sp.lot_id < 0 && !sp.is_on_lane_one) {
            add_oriented_box(sp.center + left * 0.55f, sp.heading, sp.length + 1.6f, sp.width + 1.8f,
                             -0.04f, 0.00f, {0.16f, 0.175f, 0.205f}, 1.0f, 4.0f);
        }

        bool is_target = (sp.id == tel.target_parking_space_id);
        bool is_candidate = (tel.autopark_active && tel.target_parking_space_id < 0 && sp.is_visible_to_ego && !sp.is_occupied);

        Color3 border_col = is_target    ? Color3{0.15f, 0.98f, 0.58f} :
                            is_candidate ? Color3{0.25f, 0.84f, 0.98f} :
                            sp.is_occupied ? Color3{0.68f, 0.72f, 0.78f} :
                                             Color3{0.94f, 0.97f, 1.00f};
        float line_w = is_target ? 0.19f : 0.14f;

        // 1. 4 Painted Rectangular Boundary Lines of the Parking Stall [sp.length x sp.width] (z = +0.056m)
        add_oriented_box(sp.center + left * hw, sp.heading, sp.length, line_w, 0.005f, 0.056f, border_col, 0.96f, 2.0f);
        add_oriented_box(sp.center - left * hw, sp.heading, sp.length, line_w, 0.005f, 0.056f, border_col, 0.96f, 2.0f);
        add_oriented_box(sp.center + fwd * hl,  sp.heading, line_w, sp.width,  0.005f, 0.056f, border_col, 0.96f, 2.0f);
        add_oriented_box(sp.center - fwd * hl,  sp.heading, line_w, sp.width,  0.005f, 0.056f, border_col, 0.96f, 2.0f);

        // 2. Painted "P" Symbol & Entry Orientation Chevron on the Asphalt inside Vacant Stalls
        if (!sp.is_occupied) {
            Color3 p_col = is_target ? Color3{0.18f, 1.00f, 0.62f} : Color3{0.35f, 0.76f, 0.98f};
            // Subtle translucent ground pad highlight inside vacant stall
            if (is_target || is_candidate) {
                add_oriented_box(sp.center, sp.heading, sp.length - 0.30f, sp.width - 0.30f,
                                 0.004f, 0.048f, p_col, is_target ? 0.28f : 0.14f, 2.0f);
            }
            // 3D Block Letter 'P' oriented along sp.heading (top of 'P' towards +fwd)
            add_oriented_box(sp.center + left * 0.26f, sp.heading, 1.35f, 0.13f, 0.006f, 0.058f, p_col, 0.95f, 2.0f);
            add_oriented_box(sp.center + fwd * 0.60f,  sp.heading, 0.13f, 0.62f, 0.006f, 0.058f, p_col, 0.95f, 2.0f);
            add_oriented_box(sp.center + fwd * 0.04f,  sp.heading, 0.13f, 0.62f, 0.006f, 0.058f, p_col, 0.95f, 2.0f);
            add_oriented_box(sp.center + fwd * 0.32f - left * 0.25f, sp.heading, 0.68f, 0.13f, 0.006f, 0.058f, p_col, 0.95f, 2.0f);
        }

        // Highlight target stall with a 3D bounding box
        if (!is_sensor_camera && (is_target || is_candidate)) {
            add_3d_wire_box(sp.center, sp.heading, sp.length, sp.width, 0.06f, is_target ? 1.45f : 0.65f,
                            is_target ? 0.065f : 0.04f, border_col, is_target ? 0.92f : 0.55f);
        }
    }

    // Traffic police officer directing the intersection
    void add_police_officer_3d(const Intersection& inter, bool is_visible_to_ego, bool show_perception_box) {
        Vec2 pos = inter.center;
        float facing = inter.get_police_officer_heading();
        Vec2 fwd = Vec2::from_angle(facing);
        Vec2 left = fwd.perp_left();

        add_smooth_sphere({pos.x, pos.y, 0.14f}, 0.85f, 0.85f, 0.14f, {0.22f, 0.25f, 0.30f}, 4, 12, 0.0f);
        add_smooth_sphere({pos.x, pos.y, 0.26f}, 0.72f, 0.72f, 0.05f, {0.92f, 0.94f, 0.98f}, 4, 12, 2.0f);

        float z_base = 0.28f;
        Color3 navy = {0.09f, 0.14f, 0.28f};
        add_oriented_box(pos + left * 0.11f, facing, 0.20f, 0.16f, z_base, z_base + 0.76f, navy);
        add_oriented_box(pos - left * 0.11f, facing, 0.20f, 0.16f, z_base, z_base + 0.76f, navy);

        add_tapered_oriented_box(pos, pos, facing, 0.28f, 0.44f, 0.26f, 0.48f, z_base + 0.76f, z_base + 1.38f, navy);
        add_tapered_oriented_box(pos, pos, facing, 0.30f, 0.45f, 0.28f, 0.46f, z_base + 0.86f, z_base + 1.30f, {0.78f, 0.98f, 0.12f}, 1.0f, 2.0f);
        add_oriented_box(pos, facing, 0.31f, 0.46f, z_base + 0.98f, z_base + 1.05f, {0.92f, 0.96f, 1.0f}, 1.0f, 2.0f);

        add_smooth_sphere({pos.x, pos.y, z_base + 1.54f}, 0.15f, 0.15f, 0.16f, {0.92f, 0.76f, 0.64f}, 5, 8, 0.0f);
        add_smooth_sphere({pos.x, pos.y, z_base + 1.66f}, 0.19f, 0.19f, 0.07f, {0.95f, 0.97f, 1.00f}, 4, 10, 1.0f);
        add_oriented_box(pos + fwd * 0.12f, facing, 0.16f, 0.26f, z_base + 1.60f, z_base + 1.64f, {0.06f, 0.09f, 0.18f}, 1.0f, 1.0f);

        bool is_clearance = (inter.phase == 1 || inter.phase == 2 || inter.phase == 4 || inter.phase == 5);
        if (is_clearance) {
            Vec2 l_sh = pos + left * 0.30f;
            Vec2 r_sh = pos - left * 0.30f;
            add_oriented_box(l_sh, facing, 0.14f, 0.14f, z_base + 1.28f, z_base + 1.88f, navy);
            add_oriented_box(r_sh, facing, 0.14f, 0.14f, z_base + 1.28f, z_base + 1.88f, navy);
            add_smooth_sphere({l_sh.x, l_sh.y, z_base + 1.92f}, 0.09f, 0.09f, 0.09f, {0.98f, 0.98f, 1.0f}, 4, 6, 2.0f);
            add_box({r_sh.x - 0.05f, r_sh.y - 0.05f, z_base + 1.90f},
                    {r_sh.x + 0.05f, r_sh.y + 0.05f, z_base + 2.32f}, {1.0f, 0.15f, 0.12f}, 1.0f, 2.0f);
        } else {
            Vec2 stop_arm = pos + left * 0.28f + fwd * 0.28f;
            add_oriented_box(stop_arm, facing, 0.48f, 0.13f, z_base + 1.24f, z_base + 1.38f, navy);
            Vec2 stop_glove = pos + left * 0.28f + fwd * 0.56f;
            add_oriented_box(stop_glove, facing, 0.08f, 0.18f, z_base + 1.24f, z_base + 1.48f, {1.0f, 0.20f, 0.18f}, 1.0f, 2.0f);

            float wave = std::sin(inter.officer_anim_phase) * 0.55f;
            float wave_ang = facing - PI * 0.5f + wave;
            Vec2 wave_dir = Vec2::from_angle(wave_ang);
            Vec2 r_sh = pos - left * 0.28f;
            Vec2 wave_mid = r_sh + wave_dir * 0.26f;
            add_oriented_box(wave_mid, wave_ang, 0.50f, 0.13f, z_base + 1.24f, z_base + 1.38f, navy);

            Vec2 baton_xy = r_sh + wave_dir * 0.54f;
            add_box({baton_xy.x - 0.06f, baton_xy.y - 0.06f, z_base + 1.26f},
                    {baton_xy.x + 0.06f, baton_xy.y + 0.06f, z_base + 1.72f}, {0.12f, 1.0f, 0.48f}, 1.0f, 2.0f);
        }

        if (show_perception_box && is_visible_to_ego) {
            add_3d_wire_box(pos, facing, 1.15f, 1.15f, 0.05f, 2.35f, 0.05f, {0.22f, 0.88f, 1.0f}, 0.80f);
        }
    }

    // Two-lane circular roundabout island and ring markings
    void add_roundabout_island_3d(const Intersection& inter) {
        Vec2 c = inter.center;
        float r_outer = 3.45f; // Central granite curb island radius
        float r_inner = 3.02f; // Central lush grass mound radius

        // 1. Central Landscaped Granite Curb & Grass Island
        add_smooth_sphere({c.x, c.y, 0.06f}, r_outer, r_outer, 0.14f, {0.46f, 0.48f, 0.52f}, 5, 22, 0.0f);
        add_smooth_sphere({c.x, c.y, 0.12f}, r_inner, r_inner, 0.13f, {0.18f, 0.44f, 0.24f}, 5, 22, 0.0f);
        for (int i = 0; i < 12; ++i) {
            float a = (TWO_PI * i) / 12.0f;
            Vec2 rp = c + Vec2{std::cos(a), std::sin(a)} * (r_outer - 0.18f);
            add_oriented_box(rp, a + PI * 0.5f, 0.42f, 0.14f, 0.08f, 0.24f, {0.98f, 0.76f, 0.12f}, 1.0f, 2.0f);
        }
        if (!inter.is_police_controlled()) {
            add_box({c.x - 0.18f, c.y - 0.18f, 0.15f}, {c.x + 0.18f, c.y + 0.18f, 2.05f}, {0.34f, 0.24f, 0.16f}, 1.0f, 0.0f);
            add_smooth_sphere({c.x, c.y, 2.95f}, 1.35f, 1.35f, 1.45f, {0.22f, 0.52f, 0.28f}, 6, 12, 0.0f);
        }

        // 2. Continuous Solid Yellow Inner Ring Marking around the Central Island (r = 3.80m)
        constexpr int N_INNER_SEG = 36;
        for (int i = 0; i < N_INNER_SEG; ++i) {
            float a = (TWO_PI * (i + 0.5f)) / N_INNER_SEG;
            float seg_len = (TWO_PI * 3.80f) / N_INNER_SEG + 0.04f;
            Vec2 p = c + Vec2{std::cos(a), std::sin(a)} * 3.80f;
            add_oriented_box(p, a + PI * 0.5f, seg_len, 0.14f, 0.01f, 0.055f, {0.98f, 0.78f, 0.12f}, 1.0f, 2.0f);
        }

        // 3. Circular Dashed White Lane Divider Ring (r = 6.42m) separating Roundabout Lane 2 (Inner, r=5.15m)
        //    from Roundabout Lane 1 (Outer, r=7.75m)
        constexpr int N_DIV_DASH = 24;
        for (int i = 0; i < N_DIV_DASH; ++i) {
            float a = (TWO_PI * i) / N_DIV_DASH;
            Vec2 p = c + Vec2{std::cos(a), std::sin(a)} * 6.42f;
            add_oriented_box(p, a + PI * 0.5f, 0.92f, 0.16f, 0.01f, 0.055f, {0.94f, 0.96f, 1.00f}, 0.96f, 2.0f);
        }

        // 4. Circular Dashed White Outer Yield / Perimeter Ring (r = 9.15m)
        constexpr int N_YIELD_DASH = 36;
        for (int i = 0; i < N_YIELD_DASH; ++i) {
            float a = (TWO_PI * i) / N_YIELD_DASH;
            Vec2 p = c + Vec2{std::cos(a), std::sin(a)} * 9.15f;
            add_oriented_box(p, a + PI * 0.5f, 0.72f, 0.18f, 0.01f, 0.055f, {0.92f, 0.95f, 0.99f}, 0.90f, 2.0f);
        }

        // 5. Counter-Clockwise Painted Directional Pavement Arrows inside Both Roundabout Lanes
        for (int i = 0; i < 4; ++i) {
            float a_in = (TWO_PI * i) / 4.0f + 0.38f;
            Vec2 p_in = c + Vec2{std::cos(a_in), std::sin(a_in)} * 5.15f;
            add_pavement_arrow_3d(p_in, a_in + PI * 0.5f);
        }
        for (int i = 0; i < 6; ++i) {
            float a_out = (TWO_PI * i) / 6.0f + 0.25f;
            Vec2 p_out = c + Vec2{std::cos(a_out), std::sin(a_out)} * 7.75f;
            add_pavement_arrow_3d(p_out, a_out + PI * 0.5f);
        }
    }

    void add_sensor_360_ring(const CityWorld& world) {
        const Vehicle& ego = world.ego;
        constexpr int SEG = 48;
        for (int i = 0; i < SEG; ++i) {
            float rel_a0 = (TWO_PI * i) / SEG - PI;
            float rel_a1 = (TWO_PI * (i + 1)) / SEG - PI;
            float r0 = CityWorld::get_sensor_360_range_at_angle(rel_a0);
            float r1 = CityWorld::get_sensor_360_range_at_angle(rel_a1);

            float w_a0 = ego.heading + rel_a0;
            float w_a1 = ego.heading + rel_a1;
            Vec2 p0 = ego.pos + Vec2{std::cos(w_a0), std::sin(w_a0)} * r0;
            Vec2 p1 = ego.pos + Vec2{std::cos(w_a1), std::sin(w_a1)} * r1;

            Vec2 mid = (p0 + p1) * 0.5f;
            float seg_len = (p1 - p0).len();
            float seg_ang = std::atan2(p1.y - p0.y, p1.x - p0.x);
            add_oriented_box(mid, seg_ang, seg_len, 0.18f, 0.04f, 0.06f, {0.22f, 0.65f, 0.98f}, 0.28f, 2.0f);
        }
    }

    void add_stop_sign_3d(Vec2 pos, float facing_heading) {
        add_box({pos.x - 0.05f, pos.y - 0.05f, 0.0f}, {pos.x + 0.05f, pos.y + 0.05f, 2.35f}, {0.55f, 0.58f, 0.64f}, 1.0f, 1.0f);

        Vec2 fwd = Vec2::from_angle(facing_heading);
        Vec2 left = fwd.perp_left();
        Vec3 center = {pos.x, pos.y, 2.15f};
        Vec3 n_front = {fwd.x, fwd.y, 0.0f};
        Vec3 n_back  = {-fwd.x, -fwd.y, 0.0f};

        auto draw_octagon = [&](float offset_fwd, float radius, Color3 col, Vec3 n, float mat) {
            Vec3 c = center + n_front * offset_fwd;
            for (int i = 0; i < 8; ++i) {
                float a0 = (PI * 0.125f) + (TWO_PI * i) / 8.0f;
                float a1 = (PI * 0.125f) + (TWO_PI * (i + 1)) / 8.0f;
                Vec3 p0 = c + Vec3{left.x * std::cos(a0), left.y * std::cos(a0), std::sin(a0)} * radius;
                Vec3 p1 = c + Vec3{left.x * std::cos(a1), left.y * std::cos(a1), std::sin(a1)} * radius;
                push_tri(c, p0, p1, n, n, n, col, 1.0f, mat);
            }
        };

        draw_octagon(-0.02f, 0.44f, {0.42f, 0.45f, 0.50f}, n_back,  1.0f);
        draw_octagon( 0.02f, 0.44f, {0.96f, 0.98f, 1.00f}, n_front, 2.0f);
        draw_octagon( 0.03f, 0.38f, {0.92f, 0.12f, 0.12f}, n_front, 2.0f);
        add_oriented_box(pos + fwd * 0.04f, facing_heading, 0.02f, 0.50f, 2.07f, 2.23f, {0.98f, 0.98f, 1.0f}, 1.0f, 2.0f);
    }

    void add_yield_sign_3d(Vec2 pos, float facing_heading) {
        add_box({pos.x - 0.05f, pos.y - 0.05f, 0.0f}, {pos.x + 0.05f, pos.y + 0.05f, 2.35f}, {0.55f, 0.58f, 0.64f}, 1.0f, 1.0f);

        Vec2 fwd = Vec2::from_angle(facing_heading);
        Vec2 left = fwd.perp_left();
        Vec3 n_front = {fwd.x, fwd.y, 0.0f};

        auto draw_yield_tri = [&](float offset_fwd, float scale, Color3 col) {
            Vec3 c = Vec3{pos.x, pos.y, 2.18f} + n_front * offset_fwd;
            Vec3 top_l = c + Vec3{left.x * 0.44f * scale,  left.y * 0.44f * scale,  0.32f * scale};
            Vec3 top_r = c + Vec3{-left.x * 0.44f * scale, -left.y * 0.44f * scale, 0.32f * scale};
            Vec3 bot   = c + Vec3{0.0f, 0.0f, -0.44f * scale};
            push_tri(top_l, top_r, bot, n_front, n_front, n_front, col, 1.0f, 2.0f);
        };

        draw_yield_tri(0.02f, 1.05f, {0.96f, 0.98f, 1.00f});
        draw_yield_tri(0.03f, 0.95f, {0.94f, 0.14f, 0.12f});
        draw_yield_tri(0.04f, 0.58f, {0.98f, 0.96f, 0.92f});
    }

    // 3D Blue-and-White Rectangular ONE-WAY Sign on Steel Post
    void add_one_way_sign_3d(Vec2 pos, float road_heading) {
        add_box({pos.x - 0.05f, pos.y - 0.05f, 0.0f}, {pos.x + 0.05f, pos.y + 0.05f, 2.35f}, {0.55f, 0.58f, 0.64f}, 1.0f, 1.0f);
        // Blue sign plate + white directional arrow bar
        add_oriented_box(pos, road_heading, 0.72f, 0.08f, 1.98f, 2.32f, {0.10f, 0.44f, 0.92f}, 1.0f, 2.0f);
        add_oriented_box(pos, road_heading, 0.52f, 0.10f, 2.10f, 2.20f, {0.96f, 0.98f, 1.00f}, 1.0f, 2.0f);
    }

    void add_pedestrian_signal_post_3d(Vec2 post_xy, bool is_walk_green) {
        add_box({post_xy.x - 0.05f, post_xy.y - 0.05f, 0.0f},
                {post_xy.x + 0.05f, post_xy.y + 0.05f, 2.55f}, {0.22f, 0.24f, 0.29f}, 1.0f, 1.0f);
        add_box({post_xy.x - 0.16f, post_xy.y - 0.16f, 1.92f},
                {post_xy.x + 0.16f, post_xy.y + 0.16f, 2.52f}, {0.08f, 0.09f, 0.11f}, 1.0f, 0.0f);

        Color3 dont_walk_col = !is_walk_green ? Color3{1.00f, 0.14f, 0.12f} : Color3{0.20f, 0.06f, 0.06f};
        Color3 walk_col      =  is_walk_green ? Color3{0.15f, 1.00f, 0.52f} : Color3{0.05f, 0.18f, 0.10f};

        add_box({post_xy.x - 0.21f, post_xy.y - 0.21f, 2.24f},
                {post_xy.x + 0.21f, post_xy.y + 0.21f, 2.46f}, dont_walk_col, 1.0f, !is_walk_green ? 2.0f : 0.0f);
        add_box({post_xy.x - 0.21f, post_xy.y - 0.21f, 1.96f},
                {post_xy.x + 0.21f, post_xy.y + 0.21f, 2.18f}, walk_col,      1.0f,  is_walk_green ? 2.0f : 0.0f);
    }

    void add_bicycle_3d(const Bicycle& bike, int detected_bike_id, bool show_perception_box) {
        Vec2 fwd = Vec2::from_angle(bike.heading);
        Vec2 left = fwd.perp_left();

        add_oriented_box(bike.pos, bike.heading, 1.85f, 0.55f, 0.032f, 0.038f, {0.02f, 0.03f, 0.05f}, 0.55f, 3.0f);

        Vec2 w_front = bike.pos + fwd * 0.54f;
        Vec2 w_rear  = bike.pos - fwd * 0.54f;
        add_wheel_3d(w_front, bike.heading + bike.steer_angle, 0.33f, 0.07f, 0.33f, false);
        add_wheel_3d(w_rear,  bike.heading, 0.33f, 0.07f, 0.33f, false);

        add_oriented_box(bike.pos, bike.heading, 1.05f, 0.06f, 0.32f, 0.72f, bike.frame_color, 1.0f, 1.0f);
        add_oriented_box(w_front - fwd * 0.08f, bike.heading, 0.08f, 0.52f, 0.88f, 0.94f, {0.22f, 0.24f, 0.28f}, 1.0f, 1.0f);
        add_oriented_box(w_rear - fwd * 0.12f, bike.heading, 0.06f, 0.08f, 0.68f, 0.76f, {1.0f, 0.12f, 0.12f}, 1.0f, 2.0f);

        float p_swing = std::sin(bike.pedal_phase) * 0.14f;
        Color3 shorts = {0.14f, 0.16f, 0.22f};
        add_oriented_box(bike.pos - fwd * 0.10f + left * 0.11f, bike.heading, 0.20f, 0.13f, 0.30f + p_swing, 0.82f, shorts);
        add_oriented_box(bike.pos - fwd * 0.10f - left * 0.11f, bike.heading, 0.20f, 0.13f, 0.30f - p_swing, 0.82f, shorts);

        add_tapered_oriented_box(bike.pos - fwd * 0.12f, bike.pos + fwd * 0.14f, bike.heading,
                                 0.28f, 0.38f, 0.28f, 0.40f, 0.82f, 1.36f, bike.rider_color);
        add_oriented_box(bike.pos + fwd * 0.26f + left * 0.18f, bike.heading, 0.34f, 0.09f, 0.92f, 1.24f, bike.rider_color);
        add_oriented_box(bike.pos + fwd * 0.26f - left * 0.18f, bike.heading, 0.34f, 0.09f, 0.92f, 1.24f, bike.rider_color);

        add_smooth_sphere({bike.pos.x + fwd.x * 0.18f, bike.pos.y + fwd.y * 0.18f, 1.48f}, 0.13f, 0.13f, 0.14f, {0.92f, 0.76f, 0.64f}, 5, 8, 0.0f);
        add_smooth_sphere({bike.pos.x + fwd.x * 0.18f, bike.pos.y + fwd.y * 0.18f, 1.56f}, 0.16f, 0.15f, 0.11f, {0.96f, 0.98f, 1.00f}, 5, 8, 1.0f);

        if (show_perception_box && bike.is_visible_to_ego) {
            bool is_tracked = (bike.id == detected_bike_id);
            Color3 bcol = is_tracked ? Color3{0.78f, 0.38f, 1.0f} : Color3{0.52f, 0.42f, 0.88f};
            add_3d_wire_box(bike.pos, bike.heading, 1.92f, 0.82f, 0.06f, 1.72f, 0.045f, bcol, is_tracked ? 0.88f : 0.48f);
        }
    }

    void add_sky_dome(Vec3 center, float radius = 260.0f) {
        constexpr int SECTORS = 24;
        for (int s = 0; s < SECTORS; ++s) {
            float a0 = (TWO_PI * s) / SECTORS;
            float a1 = (TWO_PI * (s + 1)) / SECTORS;
            float x0 = center.x + std::cos(a0) * radius;
            float y0 = center.y + std::sin(a0) * radius;
            float x1 = center.x + std::cos(a1) * radius;
            float y1 = center.y + std::sin(a1) * radius;

            push_quad({x0, y0, -5.0f}, {x1, y1, -5.0f}, {x1, y1, 115.0f}, {x0, y0, 115.0f},
                      {0, 0, 1}, {0.14f, 0.20f, 0.30f}, 1.0f, 6.0f);
        }
    }

    void add_wheel_3d(Vec2 center, float wheel_heading, float radius, float width, float z_center, bool is_right_side) {
        Vec2 fwd = Vec2::from_angle(wheel_heading);
        Vec2 axis = fwd.perp_left() * (is_right_side ? -1.0f : 1.0f);
        constexpr int SEG = 18;

        Vec2 p_in  = center - axis * (width * 0.5f);
        Vec2 p_out = center + axis * (width * 0.5f);
        Color3 tire_col = {0.10f, 0.11f, 0.13f};

        for (int i = 0; i < SEG; ++i) {
            float a0 = (TWO_PI * i) / SEG;
            float a1 = (TWO_PI * (i + 1)) / SEG;
            float c0 = std::cos(a0), s0 = std::sin(a0);
            float c1 = std::cos(a1), s1 = std::sin(a1);

            Vec3 n0 = {fwd.x * c0, fwd.y * c0, s0};
            Vec3 n1 = {fwd.x * c1, fwd.y * c1, s1};

            Vec3 v00 = {p_in.x  + fwd.x * c0 * radius, p_in.y  + fwd.y * c0 * radius, z_center + s0 * radius};
            Vec3 v10 = {p_out.x + fwd.x * c0 * radius, p_out.y + fwd.y * c0 * radius, z_center + s0 * radius};
            Vec3 v11 = {p_out.x + fwd.x * c1 * radius, p_out.y + fwd.y * c1 * radius, z_center + s1 * radius};
            Vec3 v01 = {p_in.x  + fwd.x * c1 * radius, p_in.y  + fwd.y * c1 * radius, z_center + s1 * radius};

            push_smooth_quad(v00, v10, v11, v01, n0, n0, n1, n1, tire_col, 1.0f, 0.0f);

            float rim_r = radius * 0.70f;
            Vec3 r0 = {p_out.x + fwd.x * c0 * rim_r, p_out.y + fwd.y * c0 * rim_r, z_center + s0 * rim_r};
            Vec3 r1 = {p_out.x + fwd.x * c1 * rim_r, p_out.y + fwd.y * c1 * rim_r, z_center + s1 * rim_r};
            Vec3 rc = {p_out.x, p_out.y, z_center};
            Vec3 n_out = {axis.x, axis.y, 0.0f};
            Color3 rim_col = (i % 3 == 0) ? Color3{0.76f, 0.80f, 0.86f} : Color3{0.26f, 0.29f, 0.34f};
            push_tri(rc, r0, r1, n_out, n_out, n_out, rim_col, 1.0f, 1.0f);
        }
    }

    // 3D City Transit Bus Mesh (9.6m long, 6 wheels, panoramic passenger windows, LED destination board)
    void add_transit_bus_3d(const Vehicle& bus, float sim_time, const RuleTelemetry& tel, bool show_perception_boxes) {
        Vec2 fwd = Vec2::from_angle(bus.heading);
        Vec2 left = fwd.perp_left();
        float L = bus.length; // 9.60m
        float W = bus.width;  // 2.46m

        // Soft Ground Shadow
        Vec2 sf = fwd * (L * 0.56f);
        Vec2 sl = left * (W * 0.62f);
        push_quad({bus.pos.x - sf.x - sl.x, bus.pos.y - sf.y - sl.y, 0.032f},
                  {bus.pos.x + sf.x - sl.x, bus.pos.y + sf.y - sl.y, 0.032f},
                  {bus.pos.x + sf.x + sl.x, bus.pos.y + sf.y + sl.y, 0.032f},
                  {bus.pos.x - sf.x + sl.x, bus.pos.y - sf.y + sl.y, 0.032f},
                  {0, 0, 1}, {0.02f, 0.03f, 0.05f}, 0.72f, 3.0f);

        // 6 Heavy-Duty Bus Wheels (Steered front axle + tandem rear axles)
        float wrad = 0.44f;
        float wy = W * 0.48f;
        add_wheel_3d(bus.pos + fwd * 2.95f + left * wy, bus.heading + bus.steer_angle, wrad, 0.28f, wrad, false);
        add_wheel_3d(bus.pos + fwd * 2.95f - left * wy, bus.heading + bus.steer_angle, wrad, 0.28f, wrad, true);
        add_wheel_3d(bus.pos - fwd * 2.15f + left * wy, bus.heading, wrad, 0.28f, wrad, false);
        add_wheel_3d(bus.pos - fwd * 2.15f - left * wy, bus.heading, wrad, 0.28f, wrad, true);
        add_wheel_3d(bus.pos - fwd * 3.25f + left * wy, bus.heading, wrad, 0.28f, wrad, false);
        add_wheel_3d(bus.pos - fwd * 3.25f - left * wy, bus.heading, wrad, 0.28f, wrad, true);

        // Lower Bus Chassis (Transit Blue/Emerald Livery)
        add_oriented_box(bus.pos, bus.heading, L, W, 0.26f, 1.32f, bus.body_color, 1.0f, 1.0f);
        // White Transit Stripe Beltline
        add_oriented_box(bus.pos, bus.heading, L * 1.002f, W * 1.005f, 1.18f, 1.34f, {0.94f, 0.96f, 0.99f}, 1.0f, 1.0f);
        // Panoramic Dark Glass Passenger Window Section (`z = 1.34m..2.52m`)
        add_oriented_box(bus.pos, bus.heading, L * 0.98f, W * 0.97f, 1.34f, 2.52f, {0.10f, 0.14f, 0.20f}, 1.0f, 1.0f);
        // Upper Roof Cap + White Roof Battery/HVAC Pods (`z = 2.52m..3.05m`)
        add_tapered_oriented_box(bus.pos, bus.pos, bus.heading, L * 0.98f, W * 0.97f, L * 0.94f, W * 0.88f,
                                 2.52f, 2.86f, {0.92f, 0.95f, 0.98f}, 1.0f, 1.0f);
        add_oriented_box(bus.pos - fwd * 0.8f, bus.heading, L * 0.55f, W * 0.62f, 2.86f, 3.06f, {0.84f, 0.88f, 0.92f}, 1.0f, 1.0f);

        // Illuminated Amber LED Front Destination Headsign ("BUS")
        Vec2 nose = bus.pos + fwd * (L * 0.492f);
        add_oriented_box(nose, bus.heading, 0.10f, W * 0.72f, 2.48f, 2.74f, {1.0f, 0.72f, 0.08f}, 1.0f, 2.0f);
        // LED Headlights
        add_oriented_box(nose + left * (W * 0.35f), bus.heading, 0.10f, 0.38f, 0.62f, 0.80f, {0.95f, 0.98f, 1.0f}, 1.0f, 2.0f);
        add_oriented_box(nose - left * (W * 0.35f), bus.heading, 0.10f, 0.38f, 0.62f, 0.80f, {0.95f, 0.98f, 1.0f}, 1.0f, 2.0f);

        // Rear Brake Lights & Boarding Hazard Indicators
        Vec2 tail = bus.pos - fwd * (L * 0.492f);
        Color3 brake_col = (bus.braking_lights || bus.is_stopped_at_bus_zone) ? Color3{1.0f, 0.12f, 0.10f} : Color3{0.68f, 0.08f, 0.10f};
        add_oriented_box(tail, bus.heading, 0.10f, W * 0.82f, 0.72f, 0.92f, brake_col, 1.0f, 2.0f);

        bool blink_on = (std::fmod(sim_time * 3.2f, 1.0f) < 0.52f);
        if (bus.is_stopped_at_bus_zone && blink_on) {
            // Flashing amber boarding lights on all 4 corners when stopped at a "BUS" zone
            Color3 amber = {1.0f, 0.72f, 0.05f};
            add_oriented_box(nose - left * (W * 0.44f), bus.heading, 0.14f, 0.18f, 0.86f, 1.05f, amber, 1.0f, 2.0f);
            add_oriented_box(tail - left * (W * 0.44f), bus.heading, 0.14f, 0.18f, 0.96f, 1.18f, amber, 1.0f, 2.0f);
            add_oriented_box(tail + left * (W * 0.44f), bus.heading, 0.14f, 0.18f, 0.96f, 1.18f, amber, 1.0f, 2.0f);
        }

        if (show_perception_boxes && bus.is_visible_to_ego) {
            bool is_lead = (bus.id == tel.detected_lead_car_id);
            Color3 bcol = bus.is_stopped_at_bus_zone ? Color3{0.98f, 0.76f, 0.12f}
                        : (is_lead ? Color3{0.25f, 0.88f, 0.98f} : Color3{0.35f, 0.55f, 0.82f});
            add_3d_wire_box(bus.pos, bus.heading, L + 0.50f, W + 0.40f, 0.08f, 3.12f, 0.065f, bcol, is_lead ? 0.88f : 0.48f);
        }
    }

    void add_sculpted_vehicle(const Vehicle& car, float sim_time, const RuleTelemetry& tel,
                              Vec2 ego_pos, bool show_perception_boxes) {
        if (car.is_bus) {
            add_transit_bus_3d(car, sim_time, tel, show_perception_boxes);
            return;
        }

        Vec2 fwd = Vec2::from_angle(car.heading);
        Vec2 left = fwd.perp_left();
        float L = car.length;
        float W = car.width;

        Vec2 sh_c = car.pos;
        Vec2 sf = fwd * (L * 0.62f);
        Vec2 sl = left * (W * 0.68f);
        push_quad({sh_c.x - sf.x - sl.x, sh_c.y - sf.y - sl.y, 0.032f},
                  {sh_c.x + sf.x - sl.x, sh_c.y + sf.y - sl.y, 0.032f},
                  {sh_c.x + sf.x + sl.x, sh_c.y + sf.y + sl.y, 0.032f},
                  {sh_c.x - sf.x + sl.x, sh_c.y - sf.y + sl.y, 0.032f},
                  {0, 0, 1}, {0.02f, 0.03f, 0.05f}, 0.68f, 3.0f);

        float wx_f = car.wheelbase * 0.50f;
        float wx_r = -car.wheelbase * 0.50f;
        float wy   = W * 0.47f;
        float wrad = 0.35f;
        add_wheel_3d(car.pos + fwd * wx_f + left * wy, car.heading + car.steer_angle, wrad, 0.25f, wrad, false);
        add_wheel_3d(car.pos + fwd * wx_f - left * wy, car.heading + car.steer_angle, wrad, 0.25f, wrad, true);
        add_wheel_3d(car.pos + fwd * wx_r + left * wy, car.heading, wrad, 0.25f, wrad, false);
        add_wheel_3d(car.pos + fwd * wx_r - left * wy, car.heading, wrad, 0.25f, wrad, true);

        struct ProfileSlice {
            float x_norm;
            float z_bot;
            float z_mid;
            float z_top;
            float w_mid;
            float w_top;
            bool glass_top;
        };
        constexpr int NUM_SLICES = 9;
        const ProfileSlice slices[NUM_SLICES] = {
            {-0.50f, 0.26f, 0.58f, 0.82f, 0.86f, 0.76f, false},
            {-0.43f, 0.22f, 0.66f, 0.96f, 0.96f, 0.84f, false},
            {-0.30f, 0.20f, 0.86f, 1.24f, 1.00f, 0.78f, true },
            {-0.12f, 0.20f, 0.88f, 1.44f, 1.00f, 0.74f, true },
            { 0.08f, 0.20f, 0.88f, 1.40f, 1.00f, 0.75f, true },
            { 0.24f, 0.20f, 0.86f, 0.90f, 0.98f, 0.84f, false},
            { 0.38f, 0.21f, 0.72f, 0.78f, 0.95f, 0.82f, false},
            { 0.47f, 0.22f, 0.56f, 0.65f, 0.90f, 0.76f, false},
            { 0.50f, 0.24f, 0.44f, 0.52f, 0.80f, 0.66f, false}
        };

        Color3 paint = car.body_color;
        Color3 glass = {0.10f, 0.14f, 0.20f};

        for (int i = 0; i < NUM_SLICES - 1; ++i) {
            const auto& s0 = slices[i];
            const auto& s1 = slices[i + 1];

            Vec2 c0 = car.pos + fwd * (s0.x_norm * L);
            Vec2 c1 = car.pos + fwd * (s1.x_norm * L);

            float hw_m0 = (W * 0.5f) * s0.w_mid;
            float hw_m1 = (W * 0.5f) * s1.w_mid;
            float hw_t0 = (W * 0.5f) * s0.w_top;
            float hw_t1 = (W * 0.5f) * s1.w_top;

            float dx = (s1.x_norm - s0.x_norm) * L;
            float dz_top = s1.z_top - s0.z_top;
            Vec3 n_roof = Vec3{-fwd.x * dz_top, -fwd.y * dz_top, dx}.normalized();
            Vec3 n_left  = Vec3{left.x, left.y, 0.22f}.normalized();
            Vec3 n_right = Vec3{-left.x, -left.y, 0.22f}.normalized();
            Vec3 n_uleft = Vec3{left.x, left.y, 0.48f}.normalized();
            Vec3 n_uright= Vec3{-left.x, -left.y, 0.48f}.normalized();

            Color3 top_col = (s0.glass_top && s1.glass_top) ? glass : paint;
            Vec3 tl0 = {c0.x + left.x * hw_t0, c0.y + left.y * hw_t0, s0.z_top};
            Vec3 tr0 = {c0.x - left.x * hw_t0, c0.y - left.y * hw_t0, s0.z_top};
            Vec3 tr1 = {c1.x - left.x * hw_t1, c1.y - left.y * hw_t1, s1.z_top};
            Vec3 tl1 = {c1.x + left.x * hw_t1, c1.y + left.y * hw_t1, s1.z_top};
            push_smooth_quad(tr0, tr1, tl1, tl0, n_roof, n_roof, n_roof, n_roof, top_col, 1.0f, 1.0f);

            Vec3 ml0 = {c0.x + left.x * hw_m0, c0.y + left.y * hw_m0, s0.z_mid};
            Vec3 mr0 = {c0.x - left.x * hw_m0, c0.y - left.y * hw_m0, s0.z_mid};
            Vec3 mr1 = {c1.x - left.x * hw_m1, c1.y - left.y * hw_m1, s1.z_mid};
            Vec3 ml1 = {c1.x + left.x * hw_m1, c1.y + left.y * hw_m1, s1.z_mid};
            push_smooth_quad(ml0, ml1, tl1, tl0, n_left, n_left, n_uleft, n_uleft, top_col, 1.0f, 1.0f);
            push_smooth_quad(mr1, mr0, tr0, tr1, n_right, n_right, n_uright, n_uright, top_col, 1.0f, 1.0f);

            Vec3 bl0 = {c0.x + left.x * hw_m0 * 0.96f, c0.y + left.y * hw_m0 * 0.96f, s0.z_bot};
            Vec3 br0 = {c0.x - left.x * hw_m0 * 0.96f, c0.y - left.y * hw_m0 * 0.96f, s0.z_bot};
            Vec3 br1 = {c1.x - left.x * hw_m1 * 0.96f, c1.y - left.y * hw_m1 * 0.96f, s1.z_bot};
            Vec3 bl1 = {c1.x + left.x * hw_m1 * 0.96f, c1.y + left.y * hw_m1 * 0.96f, s1.z_bot};
            push_smooth_quad(bl0, bl1, ml1, ml0, n_left, n_left, n_left, n_left, paint, 1.0f, 1.0f);
            push_smooth_quad(br1, br0, mr0, mr1, n_right, n_right, n_right, n_right, paint, 1.0f, 1.0f);
        }

        add_oriented_box(car.pos + fwd * (0.495f * L), car.heading, 0.08f, W * 0.72f, 0.24f, 0.54f, paint * 0.85f, 1.0f, 1.0f);
        add_oriented_box(car.pos - fwd * (0.495f * L), car.heading, 0.08f, W * 0.80f, 0.26f, 0.82f, paint * 0.85f, 1.0f, 1.0f);

        Vec2 mir = car.pos + fwd * (0.18f * L);
        add_oriented_box(mir + left * (W * 0.53f), car.heading, 0.18f, 0.16f, 0.86f, 0.98f, paint, 1.0f, 1.0f);
        add_oriented_box(mir - left * (W * 0.53f), car.heading, 0.18f, 0.16f, 0.86f, 0.98f, paint, 1.0f, 1.0f);

        Vec2 nose = car.pos + fwd * (0.488f * L);
        add_oriented_box(nose + left * (W * 0.30f), car.heading, 0.10f, 0.36f, 0.56f, 0.68f, {0.95f, 0.98f, 1.0f}, 1.0f, 2.0f);
        add_oriented_box(nose - left * (W * 0.30f), car.heading, 0.10f, 0.36f, 0.56f, 0.68f, {0.95f, 0.98f, 1.0f}, 1.0f, 2.0f);

        Vec2 tail = car.pos - fwd * (0.490f * L);
        Color3 brake_col = car.braking_lights ? Color3{1.0f, 0.12f, 0.10f} : Color3{0.72f, 0.08f, 0.10f};
        add_oriented_box(tail, car.heading, 0.10f, W * 0.80f, 0.68f, 0.79f, brake_col, 1.0f, 2.0f);
        if (car.braking_lights) {
            add_oriented_box(car.pos - fwd * (0.34f * L), car.heading, 0.07f, 0.58f, 1.24f, 1.31f, {1.0f, 0.18f, 0.12f}, 1.0f, 2.0f);
        }
        // Bright White Reverse Backup Lights when Vehicle is in GearState::REVERSE (R)
        if (car.reversing_lights) {
            add_oriented_box(tail + left * (W * 0.24f) - fwd * 0.01f, car.heading, 0.11f, 0.22f, 0.70f, 0.78f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
            add_oriented_box(tail - left * (W * 0.24f) - fwd * 0.01f, car.heading, 0.11f, 0.22f, 0.70f, 0.78f, {0.98f, 0.99f, 1.0f}, 1.0f, 2.0f);
        }

        bool blink_on = (std::fmod(sim_time * (car.is_crashed ? 3.8f : 2.2f), 1.0f) < 0.52f);
        bool turn_left_sig  = car.is_crashed || car.is_overtaking || car.active_intent == TurnIntent::LEFT;
        bool turn_right_sig = car.is_crashed || car.active_intent == TurnIntent::RIGHT;
        if (blink_on && (turn_left_sig || turn_right_sig)) {
            Color3 amber = {1.0f, 0.72f, 0.05f};
            if (turn_left_sig) {
                add_oriented_box(nose + left * (W * 0.42f), car.heading, 0.12f, 0.15f, 0.54f, 0.67f, amber, 1.0f, 2.0f);
                add_oriented_box(tail + left * (W * 0.42f), car.heading, 0.12f, 0.15f, 0.66f, 0.80f, amber, 1.0f, 2.0f);
            }
            if (turn_right_sig) {
                add_oriented_box(nose - left * (W * 0.42f), car.heading, 0.12f, 0.15f, 0.54f, 0.67f, amber, 1.0f, 2.0f);
                add_oriented_box(tail - left * (W * 0.42f), car.heading, 0.12f, 0.15f, 0.66f, 0.80f, amber, 1.0f, 2.0f);
            }
        }

        if (car.is_crashed) {
            Vec2 hood_xy = car.pos + fwd * (0.28f * L);
            for (int sp = 0; sp < 3; ++sp) {
                float cycle = std::fmod(car.smoke_phase + sp * 0.33f, 1.0f);
                float sz = 1.0f + cycle * 2.4f;
                float sr = 0.32f + cycle * 0.55f;
                float sa = (1.0f - cycle) * 0.58f;
                Vec2 drift = {std::sin(car.smoke_phase * 1.4f + sp) * 0.25f, std::cos(car.smoke_phase + sp) * 0.25f};
                add_smooth_sphere({hood_xy.x + drift.x, hood_xy.y + drift.y, sz}, sr, sr, sr * 0.85f,
                                  {0.24f, 0.26f, 0.30f}, 5, 8, 0.0f, sa);
            }
            Vec2 warn_xy = car.pos - fwd * (0.65f * L);
            add_oriented_box(warn_xy, car.heading, 0.25f, 0.45f, 0.05f, 0.48f, {1.0f, 0.28f, 0.08f}, 1.0f, 2.0f);
        }

        if (show_perception_boxes && !car.is_ego && car.is_visible_to_ego) {
            bool is_crash = car.is_crashed;
            bool is_lead = (car.id == tel.detected_lead_car_id);
            bool is_conflict = (car.id == tel.conflict_oncoming_car_id);
            Color3 bcol = is_crash    ? Color3{1.00f, 0.20f, 0.12f} :
                          is_conflict ? Color3{1.00f, 0.68f, 0.12f} :
                          is_lead     ? Color3{0.25f, 0.88f, 0.98f} :
                                        Color3{0.35f, 0.55f, 0.82f};
            float alpha = (is_crash || is_lead || is_conflict) ? 0.92f : 0.42f;
            float thick = is_crash ? 0.075f : 0.055f;
            add_3d_wire_box(car.pos, car.heading, L + 0.45f, W + 0.35f, 0.08f, 1.62f, thick, bcol, alpha);
        }
    }

    void build_world_scene(const CityWorld& world, const AutonomyController& ctrl,
                           Vec3 cam_eye, bool is_sensor_camera, bool is_chase_mode) {
        clear();
        const Vehicle& ego = world.ego;
        const auto& tel = ctrl.expert_v11.telemetry;

        // 1. 3D Sky Horizon Dome + Landscape Ground Plane (z = -0.08m)
        add_sky_dome({0.0f, 0.0f, 0.0f}, 260.0f);
        add_box({-240.0f, -240.0f, -0.30f}, {240.0f, 240.0f, -0.08f}, {0.14f, 0.21f, 0.16f}, 1.0f, 0.0f);

        // 2. Render All Directed Road Links (Orthogonal, Diagonal 45-deg, Multi-Lane Boulevards & One-Way Streets)
        //    with Continuous Solid (No-Overtaking) vs. Dotted (Overtaking Permitted) Centerlines
        constexpr float ROAD_HALF = CityWorld::LANE_WIDTH + 0.45f; // 4.15m half-width

        for (const auto& e : world.edges) {
            if (e.is_intersection_connector) continue;

            // To avoid drawing the road asphalt twice for two-way pairs (u->v and v->u),
            // only render the shared road corridor on the primary edge (or if one-way)
            bool is_primary_corridor = e.is_one_way || (e.oppose_edge_id == -1) || (e.id < e.oppose_edge_id);
            Vec2 cu = world.intersections[e.from_node].center;
            Vec2 cv = world.intersections[e.to_node].center;
            Vec2 d = (cv - cu).normalized();
            Vec2 left = d.perp_left();
            float ang = std::atan2(d.y, d.x);

            float hu = world.intersections[e.from_node].half_size;
            float hv = world.intersections[e.to_node].half_size;
            Vec2 p_start = cu + d * hu;
            Vec2 p_end   = cv - d * hv;
            Vec2 mid = (p_start + p_end) * 0.5f;
            float len = (p_end - p_start).len();
            if (len < 2.0f) continue;

            float eff_half_w = (e.num_lanes_same_dir >= 2 && !e.is_one_way) ? (ROAD_HALF + 1.4f) : ROAD_HALF;

            if (is_primary_corridor) {
                // Concrete sidewalk under-apron (z1 = -0.01m, strictly below asphalt)
                add_oriented_box(mid, ang, len, (eff_half_w + 1.65f) * 2.0f, -0.07f, -0.01f, {0.35f, 0.37f, 0.41f}, 1.0f, 0.0f);

                // Check if either side of this road corridor has an off-street Rectangular Parking Lot or Recessed Curbside Bay
                bool right_has_lot_or_bay = false;
                bool left_has_lot_or_bay  = false;
                for (const auto& lot : world.parking_lots) {
                    if (lot.road_edge_id == e.id) right_has_lot_or_bay = true;
                    if (e.oppose_edge_id >= 0 && lot.road_edge_id == e.oppose_edge_id) left_has_lot_or_bay = true;
                }
                for (const auto& sp : world.parking_spaces) {
                    if (sp.lot_id < 0 && !sp.is_on_lane_one) {
                        if (sp.road_edge_id == e.id) right_has_lot_or_bay = true;
                        if (e.oppose_edge_id >= 0 && sp.road_edge_id == e.oppose_edge_id) left_has_lot_or_bay = true;
                    }
                }

                float curb_len = std::max(4.0f, len - 9.0f);
                if (!left_has_lot_or_bay) {
                    add_oriented_box(mid + left * (eff_half_w + 0.82f), ang, curb_len, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                } else {
                    float end_seg = std::max(3.5f, (curb_len - 28.5f) * 0.5f);
                    add_oriented_box(p_start + d * (4.5f + end_seg * 0.5f) + left * (eff_half_w + 0.82f), ang, end_seg, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                    add_oriented_box(p_end   - d * (4.5f + end_seg * 0.5f) + left * (eff_half_w + 0.82f), ang, end_seg, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                }
                if (!right_has_lot_or_bay) {
                    add_oriented_box(mid - left * (eff_half_w + 0.82f), ang, curb_len, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                } else {
                    float end_seg = std::max(3.5f, (curb_len - 28.5f) * 0.5f);
                    add_oriented_box(p_start + d * (4.5f + end_seg * 0.5f) - left * (eff_half_w + 0.82f), ang, end_seg, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                    add_oriented_box(p_end   - d * (4.5f + end_seg * 0.5f) - left * (eff_half_w + 0.82f), ang, end_seg, 1.60f, -0.04f, 0.07f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                }

                // Dark procedural asphalt road surface (top at z = 0.00m)
                add_oriented_box(mid, ang, len + 1.0f, eff_half_w * 2.0f, -0.04f, 0.00f, {0.17f, 0.18f, 0.21f}, 1.0f, 4.0f);

                // Crisp White Outer Lane Edge Lines (z = +0.052m, clearly above z=0.00m asphalt)
                add_oriented_box(mid + left * (eff_half_w - 0.24f), ang, len, 0.18f, 0.005f, 0.052f, {0.94f, 0.96f, 0.99f}, 1.0f, 2.0f);
                add_oriented_box(mid - left * (eff_half_w - 0.24f), ang, len, 0.18f, 0.005f, 0.052f, {0.94f, 0.96f, 0.99f}, 1.0f, 2.0f);

                if (e.is_one_way) {
                    // ONE-WAY STREET: Dotted White Multi-Lane Center Divider + Painted One-Way Arrows + ONE-WAY Sign
                    for (float s = 4.0f; s + 2.5f < len; s += 4.8f) {
                        Vec2 dp = p_start + d * (s + 1.2f);
                        add_oriented_box(dp, ang, 2.4f, 0.15f, 0.01f, 0.053f, {0.92f, 0.95f, 0.99f}, 1.0f, 2.0f);
                    }
                    for (float s = 12.0f; s + 8.0f < len; s += 18.0f) {
                        Vec2 ap = p_start + d * s;
                        add_pavement_arrow_3d(ap - left * 1.85f, ang);
                        add_pavement_arrow_3d(ap + left * 1.85f, ang);
                    }
                    add_one_way_sign_3d(p_start - left * (eff_half_w + 0.65f) + d * 1.5f, ang);
                } else {
                    // TWO-WAY ROAD:
                    //   - Near Intersections (s < 13.5m or s > len - 13.5m): Continuous Solid Double-Yellow Line (No Overtaking)
                    //   - Open Middle Section (s in [13.5m .. len - 13.5m]): Dotted Yellow Centerline (Overtaking Permitted)
                    float solid_len = std::min(13.5f, len * 0.25f);
                    Vec2 s_head = p_start + d * (solid_len * 0.5f);
                    Vec2 s_tail = p_end   - d * (solid_len * 0.5f);
                    Color3 yel = {0.98f, 0.78f, 0.12f};

                    // Continuous double yellow near entry & exit intersections
                    add_oriented_box(s_head + left * 0.11f, ang, solid_len, 0.12f, 0.01f, 0.053f, yel, 1.0f, 2.0f);
                    add_oriented_box(s_head - left * 0.11f, ang, solid_len, 0.12f, 0.01f, 0.053f, yel, 1.0f, 2.0f);
                    add_oriented_box(s_tail + left * 0.11f, ang, solid_len, 0.12f, 0.01f, 0.053f, yel, 1.0f, 2.0f);
                    add_oriented_box(s_tail - left * 0.11f, ang, solid_len, 0.12f, 0.01f, 0.053f, yel, 1.0f, 2.0f);

                    // Dotted yellow overtaking zone in the open middle stretch
                    for (float s = solid_len + 1.5f; s + 2.4f < len - solid_len; s += 4.6f) {
                        Vec2 dp = p_start + d * (s + 1.2f);
                        add_oriented_box(dp, ang, 2.4f, 0.16f, 0.01f, 0.053f, yel, 1.0f, 2.0f);
                    }

                    // If Multi-Lane 4-Lane Boulevard (2 lanes each direction), add dashed white lane dividers on both sides
                    if (e.num_lanes_same_dir >= 2) {
                        for (float s = 5.0f; s + 2.5f < len; s += 5.2f) {
                            Vec2 dp = p_start + d * (s + 1.2f);
                            add_oriented_box(dp + left * 2.65f, ang, 2.2f, 0.13f, 0.01f, 0.053f, {0.88f, 0.92f, 0.96f}, 0.90f, 2.0f);
                            add_oriented_box(dp - left * 2.65f, ang, 2.2f, 0.13f, 0.01f, 0.053f, {0.88f, 0.92f, 0.96f}, 0.90f, 2.0f);
                        }
                    }
                }
            }
        }

        // 3. Render Designated "BUS" Pavement Stop Zones, 3D Rectangular Parking Lot Zones & Individual Marked Parking Stalls
        for (const auto& bsz : world.bus_stops) {
            add_bus_stop_zone_3d(bsz);
        }
        for (const auto& lot : world.parking_lots) {
            add_parking_lot_zone_3d(lot);
        }
        for (const auto& sp : world.parking_spaces) {
            add_parking_space_3d(sp, tel, is_sensor_camera);
        }

        // 4. Intersections: 5/6/8-Branch Roundabout Plazas, T-Junctions, Traffic Lights, Police Officers, Stop & Yield Signs
        for (const auto& inter : world.intersections) {
            float ih = inter.half_size;

            // Intersection asphalt plaza box (at z = 0.00m)
            add_box({inter.center.x - ih - 0.5f, inter.center.y - ih - 0.5f, -0.04f},
                    {inter.center.x + ih + 0.5f, inter.center.y + ih + 0.5f,  0.00f},
                    {0.17f, 0.18f, 0.21f}, 1.0f, 4.0f);

            if (inter.shape == IntersectionShape::ROUNDABOUT_PLAZA) {
                add_roundabout_island_3d(inter);
            }
            if (inter.is_police_controlled()) {
                bool officer_vis = world.has_360_line_of_sight(inter.center);
                add_police_officer_3d(inter, officer_vis, !is_sensor_camera);
            }

            // Render crosswalks, signals & signs on each connected orthogonal arm, adapted to Multi-Lane Boulevards (5.55m),
            // Roundabouts, and 45-deg Multi-Branch Junctions so every crosswalk goes strictly from sidewalk to sidewalk
            for (int arm = 0; arm < 4; ++arm) {
                Vec2 d = dir_vec(arm);
                Vec2 right = d.perp_right();

                float arm_hw = ROAD_HALF;
                float cw_dist = ih + 1.6f;
                bool arm_connected = world.get_arm_road_info(inter.id, arm, &arm_hw, &cw_dist);

                if (!arm_connected) {
                    Vec2 curb_c = inter.center + d * (ih + 0.4f);
                    add_oriented_box(curb_c, dir_angle(arm), 1.2f, ROAD_HALF * 2.1f, -0.04f, 0.14f, {0.38f, 0.40f, 0.44f}, 1.0f, 0.0f);
                    for (int b = -3; b <= 3; ++b) {
                        Color3 bcol = (b % 2 == 0) ? Color3{0.94f, 0.16f, 0.14f} : Color3{0.96f, 0.98f, 1.0f};
                        add_oriented_box(curb_c + right * (b * 1.05f), dir_angle(arm), 0.22f, 0.98f, 0.55f, 0.92f, bcol, 1.0f, 2.0f);
                    }
                    continue;
                }

                // 1. Zebra Crosswalk Stripes spanning the full width of this road arm (-arm_hw .. +arm_hw, sidewalk to sidewalk)
                Vec2 cw_center = inter.center + d * cw_dist;
                int n_stripes = (arm_hw > 4.8f) ? 6 : 4; // 13 stripes across 4-lane boulevards (11.1m), 9 across 2-lane roads (8.3m)
                for (int s = -n_stripes; s <= n_stripes; ++s) {
                    Vec2 sp = cw_center + right * (s * 0.80f);
                    add_oriented_box(sp, dir_angle(arm), 1.75f, 0.44f, 0.01f, 0.056f, {0.94f, 0.96f, 0.99f}, 1.0f, 2.0f);
                }

                // Traffic approaching inter.center along this arm (+d) travels in direction approach_dir = (arm + 2) & 3,
                // driving on the incoming right-hand side (-right of +d)
                int approach_dir = (arm + 2) & 3;
                float stop_half_span = arm_hw * 0.50f;
                float stop_bar_w     = arm_hw - 0.36f; // Covers all incoming lanes (both Lane 1 & Lane 2 on boulevards)
                Vec2 stop_bar  = inter.center + d * (cw_dist + 1.35f) - right * stop_half_span;
                Vec2 sig_strip = inter.center + d * (cw_dist + 0.95f) - right * stop_half_span;

                if (inter.ctrl_type == IntersectionControlType::TRAFFIC_LIGHTS) {
                    LightState ls = inter.get_light_for_approach(approach_dir);
                    Color3 active_sig_col = (ls == LightState::RED)    ? Color3{1.00f, 0.12f, 0.12f} :
                                            (ls == LightState::YELLOW) ? Color3{1.00f, 0.86f, 0.06f} :
                                                                         Color3{0.08f, 1.00f, 0.34f};

                    add_oriented_box(stop_bar,  dir_angle(arm), 0.42f, stop_bar_w, 0.01f, 0.058f, {0.96f, 0.98f, 1.0f}, 1.0f, 2.0f);
                    add_oriented_box(sig_strip, dir_angle(arm), 0.36f, stop_bar_w, 0.01f, 0.062f, active_sig_col, 0.90f, 2.0f);

                    // Place signal mast pole strictly on the concrete sidewalk outside arm_hw (-right * (arm_hw + 0.85m))
                    Vec2 pole_pos = inter.center + d * (cw_dist - 0.55f) - right * (arm_hw + 0.85f);
                    Vec2 mast_end = inter.center + d * (cw_dist - 0.55f) - right * (stop_half_span - 0.5f);

                    add_box({pole_pos.x - 0.16f, pole_pos.y - 0.16f, 0.0f},
                            {pole_pos.x + 0.16f, pole_pos.y + 0.16f, 5.25f}, {0.26f, 0.29f, 0.35f}, 1.0f, 1.0f);
                    Vec2 arm_mid = (pole_pos + mast_end) * 0.5f;
                    float arm_len = (mast_end - pole_pos).len();
                    add_oriented_box(arm_mid, dir_angle((arm + 1) & 3), arm_len, 0.16f, 4.85f, 5.05f, {0.26f, 0.29f, 0.35f}, 1.0f, 1.0f);
                    add_oriented_box(arm_mid, dir_angle((arm + 1) & 3), arm_len * 0.85f, 0.22f, 5.06f, 5.20f, active_sig_col, 1.0f, 2.0f);

                    Vec2 sig_xy = inter.center + d * (cw_dist - 0.35f) - right * stop_half_span;
                    add_box({sig_xy.x - 0.30f, sig_xy.y - 0.30f, 3.28f},
                            {sig_xy.x + 0.30f, sig_xy.y + 0.30f, 4.92f}, {0.16f, 0.17f, 0.20f}, 1.0f, 0.0f);
                    add_box({sig_xy.x - 0.25f, sig_xy.y - 0.25f, 3.35f},
                            {sig_xy.x + 0.25f, sig_xy.y + 0.25f, 4.85f}, {0.07f, 0.08f, 0.10f}, 1.0f, 0.0f);

                    Color3 col_r = (ls == LightState::RED)    ? Color3{1.00f, 0.10f, 0.10f} : Color3{0.20f, 0.06f, 0.06f};
                    Color3 col_y = (ls == LightState::YELLOW) ? Color3{1.00f, 0.86f, 0.06f} : Color3{0.20f, 0.16f, 0.05f};
                    Color3 col_g = (ls == LightState::GREEN)  ? Color3{0.08f, 1.00f, 0.34f} : Color3{0.05f, 0.18f, 0.08f};

                    add_box({sig_xy.x - 0.36f, sig_xy.y - 0.36f, 4.38f}, {sig_xy.x + 0.36f, sig_xy.y + 0.36f, 4.72f},
                            col_r, 1.0f, (ls == LightState::RED) ? 2.0f : 0.0f);
                    add_box({sig_xy.x - 0.36f, sig_xy.y - 0.36f, 3.92f}, {sig_xy.x + 0.36f, sig_xy.y + 0.36f, 4.26f},
                            col_y, 1.0f, (ls == LightState::YELLOW) ? 2.0f : 0.0f);
                    add_box({sig_xy.x - 0.36f, sig_xy.y - 0.36f, 3.46f}, {sig_xy.x + 0.36f, sig_xy.y + 0.36f, 3.80f},
                            col_g, 1.0f, (ls == LightState::GREEN) ? 2.0f : 0.0f);

                    float active_z = (ls == LightState::RED) ? 4.55f : ((ls == LightState::YELLOW) ? 4.09f : 3.63f);
                    add_smooth_sphere({sig_xy.x, sig_xy.y, active_z}, 0.42f, 0.42f, 0.26f, active_sig_col, 5, 8, 2.0f, 1.0f);

                    add_box({pole_pos.x - 0.22f, pole_pos.y - 0.22f, 2.20f},
                            {pole_pos.x + 0.22f, pole_pos.y + 0.22f, 3.10f}, {0.08f, 0.09f, 0.11f}, 1.0f, 0.0f);
                    add_box({pole_pos.x - 0.27f, pole_pos.y - 0.27f, 2.82f}, {pole_pos.x + 0.27f, pole_pos.y + 0.27f, 3.04f},
                            col_r, 1.0f, (ls == LightState::RED) ? 2.0f : 0.0f);
                    add_box({pole_pos.x - 0.27f, pole_pos.y - 0.27f, 2.54f}, {pole_pos.x + 0.27f, pole_pos.y + 0.27f, 2.76f},
                            col_y, 1.0f, (ls == LightState::YELLOW) ? 2.0f : 0.0f);
                    add_box({pole_pos.x - 0.27f, pole_pos.y - 0.27f, 2.26f}, {pole_pos.x + 0.27f, pole_pos.y + 0.27f, 2.48f},
                            col_g, 1.0f, (ls == LightState::GREEN) ? 2.0f : 0.0f);

                    bool ped_walk_green = inter.is_pedestrian_walk_green(arm);
                    Vec2 ped_post_xy = inter.center + d * cw_dist + right * (arm_hw + 0.75f);
                    add_pedestrian_signal_post_3d(ped_post_xy, ped_walk_green);

                } else if (inter.ctrl_type == IntersectionControlType::POLICE_OFFICER) {
                    LightState ls = inter.get_light_for_approach(approach_dir);
                    Color3 gesture_col = (ls == LightState::RED)    ? Color3{1.00f, 0.14f, 0.14f} :
                                         (ls == LightState::YELLOW) ? Color3{1.00f, 0.86f, 0.06f} :
                                                                      Color3{0.08f, 1.00f, 0.38f};
                    add_oriented_box(stop_bar,  dir_angle(arm), 0.42f, stop_bar_w, 0.01f, 0.058f, {0.96f, 0.98f, 1.0f}, 1.0f, 2.0f);
                    add_oriented_box(sig_strip, dir_angle(arm), 0.30f, stop_bar_w, 0.01f, 0.062f, gesture_col, 0.85f, 2.0f);

                    bool ped_walk_green = inter.is_pedestrian_walk_green(arm);
                    Vec2 ped_post_xy = inter.center + d * cw_dist + right * (arm_hw + 0.75f);
                    add_pedestrian_signal_post_3d(ped_post_xy, ped_walk_green);

                } else if (inter.ctrl_type == IntersectionControlType::FOUR_WAY_STOP) {
                    add_oriented_box(stop_bar, dir_angle(arm), 0.48f, stop_bar_w, 0.01f, 0.058f, {0.96f, 0.98f, 1.0f}, 1.0f, 2.0f);

                    Vec2 sign_pos = inter.center + d * (cw_dist + 1.15f) - right * (arm_hw + 0.75f);
                    add_stop_sign_3d(sign_pos, dir_angle(arm));

                } else if (inter.ctrl_type == IntersectionControlType::TWO_WAY_YIELD) {
                    if (inter.has_yield_sign_for_approach(approach_dir)) {
                        int n_teeth = (arm_hw > 4.8f) ? 3 : 2;
                        for (int t = -n_teeth; t <= n_teeth; ++t) {
                            add_oriented_box(stop_bar + right * (t * 0.72f), dir_angle(arm), 0.42f, 0.38f, 0.01f, 0.058f, {0.98f, 0.82f, 0.18f}, 1.0f, 2.0f);
                        }
                        Vec2 sign_pos = inter.center + d * (cw_dist + 1.15f) - right * (arm_hw + 0.75f);
                        add_yield_sign_3d(sign_pos, dir_angle(arm));
                    }
                }
            }
        }

        // 5. Boulevard Streetlamps & Trees
        constexpr float S = CityWorld::BLOCK_SPACING;
        for (int gy = -1; gy <= CityWorld::GRID_N - 1; ++gy) {
            for (int gx = -1; gx <= CityWorld::GRID_N - 1; ++gx) {
                float cx = (gx - 0.5f) * S;
                float cy = (gy - 0.5f) * S;

                Vec2 lamp_pos = {cx, cy + (S * 0.5f - ROAD_HALF - 1.2f)};
                add_box({lamp_pos.x - 0.08f, lamp_pos.y - 0.08f, 0.0f},
                        {lamp_pos.x + 0.08f, lamp_pos.y + 0.08f, 5.4f}, {0.24f, 0.27f, 0.32f}, 1.0f, 1.0f);
                add_box({lamp_pos.x - 0.22f, lamp_pos.y - 0.45f, 5.3f},
                        {lamp_pos.x + 0.22f, lamp_pos.y + 0.45f, 5.5f}, {1.0f, 0.92f, 0.72f}, 1.0f, 2.0f);
            }
        }

        // 6. Architectural Glass-Curtain-Wall 3D Buildings (mat_type = 5)
        for (const auto& b : world.buildings) {
            float d_eye = Vec2{(b.min_pt.x + b.max_pt.x) * 0.5f - cam_eye.x,
                               (b.min_pt.y + b.max_pt.y) * 0.5f - cam_eye.y}.len();
            if (!is_sensor_camera && is_chase_mode && d_eye < 18.0f) continue;

            add_box({b.min_pt.x, b.min_pt.y, 0.0f}, {b.max_pt.x, b.max_pt.y, 2.4f}, b.color * 0.85f, 1.0f, 0.0f);
            add_box({b.min_pt.x + 0.12f, b.min_pt.y + 0.12f, 2.4f}, {b.max_pt.x - 0.12f, b.max_pt.y - 0.12f, b.height}, b.color, 1.0f, 5.0f);
            add_box({b.min_pt.x + 0.35f, b.min_pt.y + 0.35f, b.height},
                    {b.max_pt.x - 0.35f, b.max_pt.y - 0.35f, b.height + 0.45f}, b.roof_color, 1.0f, 0.0f);
        }

        // 7. Articulated 3D Walking/Sprinting Pedestrians
        for (const auto& ped : world.pedestrians) {
            Vec2 dir = (ped.end_pt - ped.start_pt).normalized();
            float heading = std::atan2(dir.y, dir.x);
            Vec2 left = dir.perp_left();
            float swing_amp = ped.is_red_light_runner ? 0.28f : 0.18f;
            float swing = ped.active_crossing ? std::sin(ped.anim_phase) * swing_amp : 0.0f;

            Color3 pants = {0.16f, 0.19f, 0.25f};
            add_oriented_box(ped.pos + dir * swing + left * 0.11f, heading, 0.22f, 0.16f, 0.0f, 0.72f, pants);
            add_oriented_box(ped.pos - dir * swing - left * 0.11f, heading, 0.22f, 0.16f, 0.0f, 0.72f, pants);
            Color3 torso_col = ped.is_red_light_runner ? Color3{1.0f, 0.20f, 0.15f} : ped.shirt_color;
            add_tapered_oriented_box(ped.pos, ped.pos + (ped.is_red_light_runner ? dir * 0.10f : Vec2{0,0}),
                                     heading, 0.28f, 0.42f, 0.26f, 0.46f, 0.72f, 1.34f, torso_col);
            add_smooth_sphere({ped.pos.x, ped.pos.y, 1.50f}, 0.15f, 0.15f, 0.16f, {0.92f, 0.76f, 0.64f}, 5, 8, 0.0f);

            if (!is_sensor_camera && ped.is_visible_to_ego) {
                if (ped.is_red_light_runner || ped.id == tel.detected_pedestrian_id) {
                    add_3d_wire_box(ped.pos, heading, 1.05f, 1.05f, 0.05f, 1.80f, 0.06f, {1.0f, 0.18f, 0.18f}, 0.95f);
                } else if (ped.active_crossing) {
                    add_3d_wire_box(ped.pos, heading, 0.85f, 0.85f, 0.05f, 1.72f, 0.04f, {0.98f, 0.75f, 0.18f}, 0.48f);
                }
            }
        }

        // 8. 3D Bicycles & Pedaling Cyclists
        for (const auto& bike : world.bicycles) {
            add_bicycle_3d(bike, tel.detected_bicycle_id, !is_sensor_camera);
        }

        // 9. Half-lane parked and stationary vehicles
        for (const auto& pk : world.parked_cars) {
            Vehicle dummy_car{};
            dummy_car.id = 800 + pk.id;
            dummy_car.is_ego = false;
            dummy_car.pos = pk.pos;
            dummy_car.heading = pk.heading;
            dummy_car.length = pk.length;
            dummy_car.width = pk.width;
            dummy_car.body_color = pk.body_color;
            dummy_car.is_visible_to_ego = false; // Draw custom parked-car bounding box below
            add_sculpted_vehicle(dummy_car, world.sim_time, tel, ego.pos, false);

            bool blink_on = (std::fmod(world.sim_time * 3.0f, 1.0f) < 0.52f);
            if (blink_on && pk.hazard_lights_on) {
                Vec2 pfwd = Vec2::from_angle(pk.heading);
                Vec2 pleft = pfwd.perp_left();
                Color3 amber = {1.0f, 0.72f, 0.05f};
                add_oriented_box(pk.pos + pfwd * (pk.length * 0.48f) + pleft * (pk.width * 0.42f), pk.heading, 0.12f, 0.15f, 0.54f, 0.67f, amber, 1.0f, 2.0f);
                add_oriented_box(pk.pos - pfwd * (pk.length * 0.48f) + pleft * (pk.width * 0.42f), pk.heading, 0.12f, 0.15f, 0.66f, 0.80f, amber, 1.0f, 2.0f);
            }

            if (!is_sensor_camera && pk.is_visible_to_ego) {
                bool is_target = (pk.id == tel.detected_parked_car_id);
                Color3 bcol = is_target ? Color3{1.0f, 0.68f, 0.12f} : Color3{0.62f, 0.56f, 0.42f};
                add_3d_wire_box(pk.pos, pk.heading, pk.length + 0.40f, pk.width + 0.30f, 0.08f, 1.56f, 0.055f, bcol, is_target ? 0.90f : 0.45f);
            }
        }

        // 10. Ego vehicle, NPC cars, and transit buses
        if (!is_sensor_camera) {
            add_sculpted_vehicle(ego, world.sim_time, tel, ego.pos, false);
        } else {
            Vec2 efwd = Vec2::from_angle(ego.heading);
            add_tapered_oriented_box(ego.pos + efwd * 1.45f, ego.pos + efwd * 1.85f, ego.heading,
                                     1.35f, ego.width * 0.88f, 0.70f, ego.width * 0.76f,
                                     0.52f, 0.82f, ego.body_color, 1.0f, 1.0f);
        }
        for (const auto& npc : world.npc_cars) {
            add_sculpted_vehicle(npc, world.sim_time, tel, ego.pos, !is_sensor_camera);
        }

        // 11. Planned trajectory ribbons and 360-degree perception range ring
        if (!is_sensor_camera) {
            add_sensor_360_ring(world);
            const auto& v11_ribbon = ctrl.expert_v11.planned_ribbon;
            if (v11_ribbon.size() >= 2) {
                for (size_t i = 0; i + 1 < v11_ribbon.size(); ++i) {
                    Vec2 l0_dir = Vec2::from_angle(v11_ribbon[i].heading).perp_left();
                    Vec2 l1_dir = Vec2::from_angle(v11_ribbon[i + 1].heading).perp_left();
                    float s0_lat = (i < 10) ? ego.current_lat_offset : lerpf(ego.current_lat_offset, 0.0f, (i - 10) / 14.0f);
                    float s1_lat = (i + 1 < 10) ? ego.current_lat_offset : lerpf(ego.current_lat_offset, 0.0f, (i + 1 - 10) / 14.0f);
                    Vec2 p0 = v11_ribbon[i].pos + l0_dir * s0_lat;
                    Vec2 p1 = v11_ribbon[i + 1].pos + l1_dir * s1_lat;
                    Vec2 l0 = l0_dir * 0.82f;
                    Vec2 l1 = l1_dir * 0.82f;
                    float fade = 1.0f - static_cast<float>(i) / static_cast<float>(v11_ribbon.size());
                    Color3 rcol = ego.is_overtaking ? Color3{0.22f, 0.82f, 0.98f} : Color3{0.24f, 0.52f, 0.96f};
                    push_quad({p0.x - l0.x, p0.y - l0.y, 0.068f}, {p1.x - l1.x, p1.y - l1.y, 0.068f},
                              {p1.x + l1.x, p1.y + l1.y, 0.068f}, {p0.x + l0.x, p0.y + l0.y, 0.068f},
                              {0, 0, 1}, rcol, 0.30f * fade, 2.0f);
                }
            }

            const auto& v12_ribbon = ctrl.cnn_predicted_ribbon;
            if (v12_ribbon.size() >= 2) {
                bool cnn_active = (ctrl.mode == DriveMode::NEURAL_V12_AUTONOMOUS || ctrl.mode == DriveMode::DAGGER_TRAINING);
                Color3 ccol = cnn_active ? Color3{0.18f, 0.88f, 0.56f} : Color3{0.68f, 0.45f, 0.95f};
                for (size_t i = 0; i + 1 < v12_ribbon.size(); ++i) {
                    Vec2 p0 = v12_ribbon[i];
                    Vec2 p1 = v12_ribbon[i + 1];
                    Vec2 tan = (p1 - p0).normalized();
                    Vec2 l = tan.perp_left() * 0.52f;
                    float fade = 1.0f - static_cast<float>(i) / static_cast<float>(v12_ribbon.size());
                    push_quad({p0.x - l.x, p0.y - l.y, 0.082f}, {p1.x - l.x, p1.y - l.y, 0.082f},
                              {p1.x + l.x, p1.y + l.y, 0.082f}, {p0.x + l.x, p0.y + l.y, 0.082f},
                              {0, 0, 1}, ccol, 0.48f * fade, 2.0f);
                }
            }
        }
    }
};

} // namespace sim
