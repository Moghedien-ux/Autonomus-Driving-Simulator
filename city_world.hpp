#pragma once
// Procedural city environment, road network graph, vehicle kinematics,
// parking zones, traffic actors, and 360-degree line-of-sight perception.
#include "math_types.hpp"
#include <vector>
#include <array>
#include <cstring>
#include <ctime>
#include <chrono>

namespace sim {

constexpr int CAM_W = 32;
constexpr int CAM_H = 24;
constexpr int CAM_CHANNELS = 3;
constexpr int NUM_CAMERAS = 2;
constexpr int TOTAL_IMG_CHANNELS = NUM_CAMERAS * CAM_CHANNELS; // 6 channels
constexpr int TOTAL_IMG_FLOATS = TOTAL_IMG_CHANNELS * CAM_W * CAM_H;

enum class TurnIntent : int {
    LEFT = 0,
    STRAIGHT = 1,
    RIGHT = 2
};

enum class LightState : int {
    GREEN = 0,
    YELLOW = 1,
    RED = 2
};

enum class IntersectionControlType : int {
    TRAFFIC_LIGHTS = 0,
    FOUR_WAY_STOP = 1,
    TWO_WAY_YIELD = 2,
    POLICE_OFFICER = 3
};

enum class IntersectionShape : int {
    CROSS_PLUS = 0,         // 4-branch '+' intersection
    T_JUNCTION = 1,         // 3-branch 'T' or 'Y' junction
    ROUNDABOUT_PLAZA = 2,   // Circular island roundabout (4 to 8 branches)
    MULTI_BRANCH_STAR = 3   // 5, 6, or 8-branch star junction with diagonal avenues
};

// 8-Compass Direction Indices (0..7 in 45-degree increments:
// 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE)
// Note: For backwards-compatible 4-cardinal queries (0=E, 1=N, 2=W, 3=S):
inline Vec2 dir_vec(int dir4) {
    switch (dir4 & 3) {
        case 0: return { 1.0f,  0.0f};
        case 1: return { 0.0f,  1.0f};
        case 2: return {-1.0f,  0.0f};
        default: return { 0.0f, -1.0f};
    }
}

inline float dir_angle(int dir4) {
    switch (dir4 & 3) {
        case 0: return 0.0f;
        case 1: return PI * 0.5f;
        case 2: return PI;
        default: return -PI * 0.5f;
    }
}

inline bool segment_intersects_aabb(Vec2 p0, Vec2 p1, Vec2 box_min, Vec2 box_max) {
    Vec2 d = p1 - p0;
    float tmin = 0.0f, tmax = 1.0f;

    auto clip_axis = [&](float p, float delta, float bmin, float bmax) -> bool {
        if (std::fabs(delta) < 1e-6f) {
            return (p >= bmin && p <= bmax);
        }
        float inv_d = 1.0f / delta;
        float t1 = (bmin - p) * inv_d;
        float t2 = (bmax - p) * inv_d;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        return tmin <= tmax;
    };

    if (!clip_axis(p0.x, d.x, box_min.x, box_max.x)) return false;
    if (!clip_axis(p0.y, d.y, box_min.y, box_max.y)) return false;
    return (tmin <= tmax && tmax >= 0.0f && tmin <= 1.0f);
}

struct PathSample {
    Vec2 pos;
    float heading = 0.0f;
    float curvature = 0.0f;
    float s = 0.0f;
    int edge_id = -1;
};

// Designated Bus Stop Zone marked with "BUS" on the pavement + 3D Bus Shelter
struct BusStopZone {
    int id = 0;
    int edge_id = -1;
    float s_center = 28.0f;
    float length = 13.5f;
    Vec2 pos;
    float heading = 0.0f;
};

enum class GearState : int {
    DRIVE = 0,
    REVERSE = 1,
    PARK = 2
};

enum class ParkingLayoutType : int {
    PARALLEL_LATERAL_REVERSE = 0, // Lateral (parallel) stall (on Lane 1 of multi-lane road or recessed curbside)
    PERPENDICULAR_90_REVERSE = 1, // 90-deg perpendicular bay inside a Rectangular Parking Lot (Reverse-In)
    PERPENDICULAR_90_FRONT   = 2, // 90-deg perpendicular bay inside a Rectangular Parking Lot (Front-In)
    ANGLED_45_FRONT          = 3  // 45-deg angled echelon bay inside a Rectangular Parking Lot (Front-In)
};

// Rectangular Off-Street Parking Lot Zone alongside a road
struct ParkingLotZone {
    int id = 0;
    int road_edge_id = -1;
    Vec2 center;            // Center of the rectangular asphalt parking lot pad
    float heading = 0.0f;   // Orientation parallel to the adjacent road edge
    float length = 23.0f;   // Rectangular lot length along road (23.0m)
    float width = 12.2f;    // Rectangular lot depth perpendicular to road (12.2m)
    Vec2 driveway_in_pt;    // Point on road/driveway where car enters the lot aisle
    Vec2 driveway_out_pt;   // Point on road/driveway where car exits the lot aisle back to road
    float exit_edge_s = 46.0f;
    ParkingLayoutType layout = ParkingLayoutType::PERPENDICULAR_90_REVERSE;
};

// Individual Marked Rectangular Parking Stall (either On-Street Lateral or inside a Rectangular Parking Lot)
struct ParkingSpace {
    int id = 0;
    int road_edge_id = -1;
    int lot_id = -1;              // -1 if On-Street Lateral (Parallel), >= 1 if inside a Rectangular ParkingLotZone
    bool is_on_lane_one = false;  // True if lateral stall is painted directly on Lane 1 of a multi-lane road
    ParkingLayoutType layout = ParkingLayoutType::PARALLEL_LATERAL_REVERSE;

    Vec2 center;                  // Exact center of the painted rectangular parking stall
    float heading = 0.0f;         // Target vehicle heading when parked inside the bounds
    float length = 5.25f;         // Rectangular stall length
    float width = 2.85f;          // Rectangular stall width
    float road_s = 30.0f;         // Longitudinal arc-length s along road_edge_id closest to this space

    Vec2 road_turn_in_pt;         // Waypoint 0: Point ON the road lane right at the driveway/row entrance (no off-road shortcuts)
    Vec2 aisle_entry_pt;          // Waypoint 1: Entry into lot aisle (or passing lane approach for lateral)
    float aisle_heading = 0.0f;
    Vec2 setup_pt;                // Waypoint 2: Pre-turn / pre-reverse setup point in aisle or passing lane
    float setup_heading = 0.0f;
    Vec2 bay_mouth_pt;            // Waypoint 2.5: Alignment point right at the mouth of the stall
    Vec2 aisle_exit_pt;           // Waypoint 3: Point along the lot aisle before turning out the exit driveway onto the road
    Vec2 exit_merge_pt;           // Waypoint 4: Merge point back onto the road edge
    float exit_road_s = 44.0f;

    bool is_occupied = false;       // True if another parked car occupies this stall
    int occupant_car_id = -1;
    bool is_visible_to_ego = false; // Updated every frame strictly via 360-degree Line-of-Sight
};

// Half-Lane Parked Vehicle / Occupied Parking Space Car
struct HalfParkedCar {
    int id = 0;
    int edge_id = -1;
    int parking_space_id = -1;    // >= 1 if parked inside a designated ParkingSpace, -1 if half-lane hazard car
    float s_pos = 35.0f;
    Vec2 pos;
    float heading = 0.0f;
    float length = 4.55f;
    float width = 1.88f;
    Color3 body_color = {0.42f, 0.46f, 0.52f};
    bool hazard_lights_on = true;
    bool is_visible_to_ego = false;
};

struct LaneEdge {
    int id = -1;
    bool is_intersection_connector = false;
    int from_node = -1;
    int to_node = -1;
    int entry_dir = 0;       // Nearest cardinal group (0=E, 1=N, 2=W, 3=S) for signal phasing
    int exit_dir = 0;
    float exact_angle = 0.0f;// Exact road angle in radians (supports 45-deg diagonal 5/6/8-branch roads)
    bool is_diagonal = false;
    bool is_one_way = false; // True if all lanes on this road link travel in the same direction
    int num_lanes_same_dir = 1; // 1 = standard, 2 = multi-lane boulevard or 2-lane one-way street
    int oppose_edge_id = -1; // ID of opposite-direction edge (-1 if one-way street)
    TurnIntent turn = TurnIntent::STRAIGHT;

    CubicBezier2D curve;
    float length = 0.0f;
    std::vector<PathSample> samples;
    std::vector<int> next_edges;

    // Continuous vs. Dotted Overtaking Line Rule:
    //   - On One-Way streets or Multi-Lane Boulevards (num_lanes_same_dir >= 2), passing in the adjacent
    //     left lane is permitted anywhere outside 10m of intersections
    //   - On 2-way roads, the center divider is CONTINUOUS (solid double line, no overtaking) within
    //     13.5m of intersections, and DOTTED (dashed overtaking zone) in the open middle [13.5m .. length - 13.5m]
    bool is_dotted_overtake_zone(float s) const {
        if (is_intersection_connector) return false;
        float margin = (is_one_way || num_lanes_same_dir >= 2) ? 9.5f : 13.5f;
        return (s >= margin && s <= (length - margin));
    }

    PathSample eval_at_s(float s_query) const {
        if (samples.empty()) return {};
        if (s_query <= 0.0f) return samples.front();
        if (s_query >= length) return samples.back();
        for (size_t i = 0; i + 1 < samples.size(); ++i) {
            if (samples[i + 1].s >= s_query) {
                float ds = std::max(1e-4f, samples[i + 1].s - samples[i].s);
                float alpha = clampf((s_query - samples[i].s) / ds, 0.0f, 1.0f);
                PathSample out;
                out.pos = samples[i].pos * (1.0f - alpha) + samples[i + 1].pos * alpha;
                float dh = wrap_angle(samples[i + 1].heading - samples[i].heading);
                out.heading = wrap_angle(samples[i].heading + dh * alpha);
                out.curvature = lerpf(samples[i].curvature, samples[i + 1].curvature, alpha);
                out.s = s_query;
                out.edge_id = id;
                return out;
            }
        }
        return samples.back();
    }
};

struct Intersection {
    int id = 0;
    int grid_x = 0;
    int grid_y = 0;
    Vec2 center;
    float half_size = 8.2f;

    IntersectionControlType ctrl_type = IntersectionControlType::TRAFFIC_LIGHTS;
    IntersectionShape shape = IntersectionShape::CROSS_PLUS;
    int closed_arm = -1;
    bool major_is_ew = true;
    int branch_count = 4; // 3, 4, 5, 6, or 8 branches
    std::vector<float> branch_angles; // Angles (radians) of all connected road arms around this node

    int phase = 0;
    float phase_timer = 0.0f;
    float green_duration = 11.0f;
    float yellow_duration = 3.2f;
    float all_red_duration = 1.2f;
    float officer_anim_phase = 0.0f;

    bool is_arm_active(int arm_dir) const {
        return (arm_dir & 3) != closed_arm;
    }

    bool has_stop_sign_for_approach(int travel_dir) const {
        return ctrl_type == IntersectionControlType::FOUR_WAY_STOP;
    }

    bool has_yield_sign_for_approach(int travel_dir) const {
        if (ctrl_type == IntersectionControlType::TWO_WAY_YIELD) {
            bool is_ew = (travel_dir == 0 || travel_dir == 2);
            return is_ew != major_is_ew;
        }
        return false;
    }

    bool is_police_controlled() const {
        return ctrl_type == IntersectionControlType::POLICE_OFFICER;
    }

    bool uses_phase_signal() const {
        return ctrl_type == IntersectionControlType::TRAFFIC_LIGHTS ||
               ctrl_type == IntersectionControlType::POLICE_OFFICER;
    }

    LightState get_light_for_approach(int travel_dir, float* out_time_remaining = nullptr) const {
        if (!uses_phase_signal()) {
            if (out_time_remaining) *out_time_remaining = 10.0f;
            return LightState::GREEN;
        }
        bool is_ew = (travel_dir == 0 || travel_dir == 2);
        if (is_ew) {
            if (phase == 0) {
                if (out_time_remaining) *out_time_remaining = green_duration - phase_timer;
                return LightState::GREEN;
            } else if (phase == 1) {
                if (out_time_remaining) *out_time_remaining = yellow_duration - phase_timer;
                return LightState::YELLOW;
            } else {
                if (out_time_remaining) *out_time_remaining = 5.0f;
                return LightState::RED;
            }
        } else {
            if (phase == 3) {
                if (out_time_remaining) *out_time_remaining = green_duration - phase_timer;
                return LightState::GREEN;
            } else if (phase == 4) {
                if (out_time_remaining) *out_time_remaining = yellow_duration - phase_timer;
                return LightState::YELLOW;
            } else {
                if (out_time_remaining) *out_time_remaining = 5.0f;
                return LightState::RED;
            }
        }
    }

    bool is_pedestrian_walk_green(int crosswalk_arm) const {
        if (!uses_phase_signal()) {
            return true;
        }
        bool crosses_ew_road = (crosswalk_arm == 0 || crosswalk_arm == 2);
        if (crosses_ew_road) {
            return (phase == 3 && phase_timer < (green_duration - 2.2f));
        } else {
            return (phase == 0 && phase_timer < (green_duration - 2.2f));
        }
    }

    float get_police_officer_heading() const {
        bool ew_flowing = (phase == 0 || phase == 1 || phase == 2);
        return ew_flowing ? (PI * 0.5f) : 0.0f;
    }

    void step(float dt) {
        officer_anim_phase += dt * 4.5f;
        if (!uses_phase_signal()) return;
        phase_timer += dt;
        float limit = green_duration;
        if (phase == 1 || phase == 4) limit = yellow_duration;
        else if (phase == 2 || phase == 5) limit = all_red_duration;

        if (phase_timer >= limit) {
            phase_timer -= limit;
            phase = (phase + 1) % 6;
        }
    }
};

struct Building3D {
    Vec2 min_pt;
    Vec2 max_pt;
    float height = 14.0f;
    Color3 color;
    Color3 roof_color;
};

struct Pedestrian {
    int id = 0;
    Vec2 pos;
    Vec2 vel;
    Vec2 start_pt;
    Vec2 end_pt;
    float progress = 0.0f;
    float base_walk_speed = 1.65f;
    float current_speed = 1.65f;
    int intersection_id = 0;
    int crosswalk_dir = 0;
    bool active_crossing = false;
    bool is_jaywalker = false;
    bool is_red_light_runner = false;
    bool is_visible_to_ego = false;
    float wait_timer = 0.0f;
    float anim_phase = 0.0f;
    Color3 shirt_color;
};

struct Bicycle {
    int id = 0;
    Vec2 pos;
    float heading = 0.0f;
    float speed = 4.2f;
    float cruise_speed = 4.5f;
    float steer_angle = 0.0f;
    float pedal_phase = 0.0f;
    float lateral_offset = 1.12f;
    bool is_visible_to_ego = false;
    Color3 frame_color = {0.15f, 0.85f, 0.95f};
    Color3 rider_color = {0.95f, 0.82f, 0.12f};

    std::vector<int> route_edges;
    int current_route_idx = 0;
    float current_edge_s = 0.0f;
};

struct Vehicle {
    int id = 0;
    bool is_ego = false;
    bool is_bus = false;             // True for City Transit Buses
    bool is_stopped_at_bus_zone = false;
    float bus_stop_dwell_timer = 0.0f;
    int last_serviced_bus_stop_id = -1;

    Vec2 pos;
    float heading = 0.0f;
    float speed = 0.0f;              // Speed magnitude (>= 0.0 m/s)
    GearState gear = GearState::DRIVE; // DRIVE (D), REVERSE (R), or PARK (P)
    float steer_angle = 0.0f;
    float steer_cmd = 0.0f;
    float accel_cmd = 0.0f;
    float actual_accel = 0.0f;
    float lat_accel = 0.0f;
    float grip_utilization = 0.0f;
    bool is_slipping = false;
    bool braking_lights = false;
    bool reversing_lights = false;

    // Overtaking / Multi-lane lateral offset (0.0 = normal right lane, -2.8m = left passing lane)
    // Note: in our coordinate convention, left of lane centerline is positive cross-track,
    // so passing on the left uses positive lateral shift along `left_perp`
    float current_lat_offset = 0.0f;
    bool is_overtaking = false;

    bool is_visible_to_ego = false;
    bool is_crashed = false;
    bool is_reckless_driver = false;
    float crash_timer = 0.0f;
    float smoke_phase = 0.0f;
    int crash_partner_id = -1;

    float wheelbase = 2.85f;
    float length = 4.65f;
    float width = 1.92f;
    Color3 body_color = {0.15f, 0.55f, 0.95f};

    std::vector<int> route_edges;
    int current_route_idx = 0;
    float current_edge_s = 0.0f;
    TurnIntent active_intent = TurnIntent::STRAIGHT;

    float intersection_wait_time = 0.0f;
    float yield_impatience = 0.0f;
    float stop_sign_hold_timer = 0.0f;
    bool stop_sign_completed_for_edge = false;

    struct SkidPoint {
        Vec2 left_wheel;
        Vec2 right_wheel;
        float alpha = 1.0f;
    };
    std::vector<SkidPoint> skid_trail;

    void step_physics(float dt) {
        if (is_crashed) {
            speed = 0.0f;
            actual_accel = 0.0f;
            lat_accel = 0.0f;
            braking_lights = true;
            smoke_phase += dt * 2.5f;
            return;
        }

        if (gear == GearState::PARK) {
            speed = 0.0f;
            actual_accel = 0.0f;
            lat_accel = 0.0f;
            braking_lights = true;
            reversing_lights = false;
            return;
        }

        float max_steer_rad = (speed < 5.8f) ? 0.68f : 0.58f;
        float steer_rate    = (speed < 5.8f) ? 3.8f  : 2.8f;
        constexpr float MAX_ACCEL = 4.8f;
        constexpr float MAX_BRAKE = 8.8f;
        constexpr float MAX_SPEED = 15.5f;
        constexpr float MAX_REV_SPEED = 3.4f;
        constexpr float MU_ROAD = 0.78f;
        constexpr float GRAVITY = 9.81f;
        constexpr float MAX_GRIP_ACCEL = MU_ROAD * GRAVITY;

        steer_cmd = clampf(steer_cmd, -1.0f, 1.0f);
        accel_cmd = clampf(accel_cmd, -1.0f, 1.0f);

        float target_steer = steer_cmd * max_steer_rad;
        float max_dsteer = steer_rate * dt;
        steer_angle += clampf(target_steer - steer_angle, -max_dsteer, max_dsteer);

        float lon_force = (accel_cmd >= 0.0f) ? (accel_cmd * MAX_ACCEL) : (accel_cmd * MAX_BRAKE);
        float drag = 0.015f * speed * std::fabs(speed) + 0.12f * speed;
        actual_accel = lon_force - drag;
        braking_lights = (accel_cmd < -0.08f);
        reversing_lights = (gear == GearState::REVERSE);

        speed += actual_accel * dt;
        if (speed < 0.0f) {
            speed = 0.0f;
            if (actual_accel < 0.0f) actual_accel = 0.0f;
        }
        float max_v = (gear == GearState::REVERSE) ? MAX_REV_SPEED : (is_bus ? 9.5f : MAX_SPEED);
        speed = clampf(speed, 0.0f, max_v);

        // Signed longitudinal velocity: positive in DRIVE, negative in REVERSE
        float signed_speed = (gear == GearState::REVERSE) ? -speed : speed;
        float kappa = std::tan(steer_angle) / wheelbase;
        float ideal_yaw_rate = signed_speed * kappa;
        lat_accel = speed * std::fabs(ideal_yaw_rate);

        float total_tire_demand = std::sqrt(lat_accel * lat_accel + (actual_accel * 0.6f) * (actual_accel * 0.6f));
        grip_utilization = total_tire_demand / MAX_GRIP_ACCEL;

        float effective_yaw_rate = ideal_yaw_rate;
        Vec2 forward = Vec2::from_angle(heading);
        Vec2 vel_vec = forward * signed_speed;

        if (std::fabs(lat_accel) > MAX_GRIP_ACCEL) {
            is_slipping = true;
            float sign = (ideal_yaw_rate >= 0.0f) ? 1.0f : -1.0f;
            float max_yaw_rate = (speed > 0.5f) ? (sign * MAX_GRIP_ACCEL / speed) : ideal_yaw_rate;
            effective_yaw_rate = lerpf(ideal_yaw_rate, max_yaw_rate, 0.78f);

            Vec2 outward = (sign >= 0.0f) ? forward.perp_right() : forward.perp_left();
            float slip_speed = clampf((std::fabs(lat_accel) - MAX_GRIP_ACCEL) * 0.18f, 0.0f, 3.0f);
            vel_vec = vel_vec + outward * slip_speed;

            Vec2 rear_center = pos - forward * (wheelbase * 0.45f);
            Vec2 right_dir = forward.perp_right();
            if (skid_trail.size() > 140) skid_trail.erase(skid_trail.begin());
            skid_trail.push_back({rear_center - right_dir * 0.8f, rear_center + right_dir * 0.8f, 1.0f});
        } else {
            is_slipping = false;
        }

        for (auto& s : skid_trail) {
            s.alpha = std::max(0.0f, s.alpha - dt * 0.15f);
        }

        pos += vel_vec * dt;
        heading = wrap_angle(heading + effective_yaw_rate * dt);

        // Physical central roundabout island curb (Node 4 at origin {0, 0}, island radius 3.45m + car half-width 0.95m = 4.40m)
        float r_from_center = pos.len();
        if (r_from_center < 4.42f && r_from_center > 0.05f) {
            Vec2 radial = pos / r_from_center;
            pos = radial * 4.42f;
        }
    }
};

class CityWorld {
public:
    static constexpr int GRID_N = 3;
    // Increased block spacing to 84.0m so straight roads have long open overtaking zones (~65m straight length)
    static constexpr float BLOCK_SPACING = 84.0f;
    static constexpr float LANE_OFFSET = 1.85f;
    static constexpr float LANE_WIDTH = 3.7f;
    static constexpr float INTER_HALF = 9.6f;

    std::vector<Intersection> intersections;
    std::vector<LaneEdge> edges;
    std::vector<Building3D> buildings;
    std::vector<BusStopZone> bus_stops;
    std::vector<ParkingLotZone> parking_lots;
    std::vector<ParkingSpace> parking_spaces;
    std::vector<HalfParkedCar> parked_cars;
    std::vector<Pedestrian> pedestrians;
    std::vector<Bicycle> bicycles;
    std::vector<Vehicle> npc_cars; // Includes standard NPC cars AND City Transit Buses (`is_bus == true`)
    Vehicle ego;

    std::array<float, TOTAL_IMG_FLOATS> camera_tensor{};

    FastRNG rng{0xC0FFEE42u};
    uint32_t city_seed = 0;
    float sim_time = 0.0f;
    float spontaneous_runner_cooldown = 3.0f;
    float spontaneous_accident_cooldown = 8.0f;
    int active_accident_count = 0;

    CityWorld() {
        auto now_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint32_t t_seed = static_cast<uint32_t>(now_ns) ^ static_cast<uint32_t>(now_ns >> 32) ^ 0x9E3779B9u;
        randomize_and_reset_city(t_seed);
    }

    int node_index(int gx, int gy) const {
        return gy * GRID_N + gx;
    }

    // Query whether orthogonal arm (0=E, 1=N, 2=W, 3=S) at intersection inter_id is connected to a road,
    // and compute its true road half-width (5.55m for 4-lane boulevards, 4.15m for 2-lane roads)
    // plus its exact crosswalk distance from inter.center (set back past any 45-deg diagonal branch or
    // roundabout apron so every crosswalk connects sidewalk-to-sidewalk and never leads into another road).
    bool get_arm_road_info(int inter_id, int arm, float* out_half_w = nullptr, float* out_cw_dist = nullptr) const {
        constexpr float BASE_HALF = LANE_WIDTH + 0.45f; // 4.15m
        bool connected = false;
        bool is_multilane_two_way = false;
        for (const auto& e : edges) {
            if (!e.is_intersection_connector && !e.is_diagonal) {
                if ((e.to_node == inter_id && ((e.exit_dir + 2) & 3) == arm) ||
                    (e.from_node == inter_id && e.entry_dir == arm)) {
                    connected = true;
                    if (e.num_lanes_same_dir >= 2 && !e.is_one_way) {
                        is_multilane_two_way = true;
                    }
                }
            }
        }
        if (!connected) return false;

        float hw = is_multilane_two_way ? (BASE_HALF + 1.40f) : BASE_HALF;
        if (out_half_w) *out_half_w = hw;

        if (out_cw_dist) {
            const auto& inter = intersections[inter_id];
            float arm_ang = dir_angle(arm);
            bool has_adj_diagonal = (inter.shape == IntersectionShape::ROUNDABOUT_PLAZA);
            for (float ba : inter.branch_angles) {
                float da = std::fabs(wrap_angle(ba - arm_ang));
                if (da > 0.20f && da < 1.15f) {
                    has_adj_diagonal = true;
                    break;
                }
            }
            // Set crosswalk back outside the intersection box (and +5.4m back when bordering a 45-deg diagonal
            // branch or roundabout so the crosswalk clears the diagonal road by >1.8m and lands on the sidewalk)
            *out_cw_dist = inter.half_size + (has_adj_diagonal ? 5.4f : 1.6f);
        }
        return true;
    }

    static float get_sensor_360_range_at_angle(float rel_angle_rad) {
        float abs_a = std::fabs(wrap_angle(rel_angle_rad));
        if (abs_a < 35.0f * DEG2RAD)  return 52.0f; // Forward main & telephoto cameras
        if (abs_a < 95.0f * DEG2RAD)  return 34.0f; // Wide & forward-side B-pillar cameras
        if (abs_a < 145.0f * DEG2RAD) return 28.0f; // Side-repeater cameras
        return 24.0f;                               // Rear camera
    }

    bool has_360_line_of_sight(Vec2 target_pos, int ignore_npc_id = -1) const {
        Vec2 rel = target_pos - ego.pos;
        float dist = rel.len();
        if (dist < 0.5f) return true;

        float abs_bearing = std::atan2(rel.y, rel.x);
        float rel_angle = wrap_angle(abs_bearing - ego.heading);
        float max_sector_range = get_sensor_360_range_at_angle(rel_angle);
        if (dist > max_sector_range) {
            return false;
        }

        for (const auto& b : buildings) {
            Vec2 bmin = {b.min_pt.x + 0.2f, b.min_pt.y + 0.2f};
            Vec2 bmax = {b.max_pt.x - 0.2f, b.max_pt.y - 0.2f};
            if (segment_intersects_aabb(ego.pos, target_pos, bmin, bmax)) {
                return false;
            }
        }

        for (const auto& npc : npc_cars) {
            if (npc.id == ignore_npc_id) continue;
            Vec2 to_npc = npc.pos - ego.pos;
            float d_npc = to_npc.len();
            if (d_npc + 2.4f >= dist || d_npc < 1.5f) continue;
            Vec2 ray_dir = rel / dist;
            float proj = to_npc.dot(ray_dir);
            if (proj > 1.5f && proj < dist - 1.8f) {
                float perp_d = std::fabs(to_npc.cross(ray_dir));
                if (perp_d < 1.05f) {
                    return false;
                }
            }
        }
        return true;
    }

    // Dual-vantage line-of-sight check for overtaking (windshield + left mirror),
    // occluded only by buildings so a vehicle ahead in Lane 1 doesn't mask oncoming traffic in Lane 2.
    bool has_360_overtake_safety_los(Vec2 target_pos) const {
        Vec2 fwd = Vec2::from_angle(ego.heading);
        Vec2 left = fwd.perp_left();
        Vec2 vantage_pts[2] = {ego.pos, ego.pos + left * 1.25f};

        Vec2 rel = target_pos - ego.pos;
        float dist = rel.len();
        if (dist < 0.5f) return true;

        float abs_bearing = std::atan2(rel.y, rel.x);
        float rel_angle = wrap_angle(abs_bearing - ego.heading);
        float max_range = (std::fabs(rel_angle) < 42.0f * DEG2RAD) ? 58.0f : get_sensor_360_range_at_angle(rel_angle);
        if (dist > max_range) return false;

        for (int v = 0; v < 2; ++v) {
            bool blocked_by_building = false;
            for (const auto& b : buildings) {
                Vec2 bmin = {b.min_pt.x + 0.15f, b.min_pt.y + 0.15f};
                Vec2 bmax = {b.max_pt.x - 0.15f, b.max_pt.y - 0.15f};
                if (segment_intersects_aabb(vantage_pts[v], target_pos, bmin, bmax)) {
                    blocked_by_building = true;
                    break;
                }
            }
            if (!blocked_by_building) return true;
        }
        return false;
    }

    void update_360_perception_visibility() {
        active_accident_count = 0;
        for (auto& npc : npc_cars) {
            npc.is_visible_to_ego = has_360_line_of_sight(npc.pos, npc.id);
            if (npc.is_crashed) ++active_accident_count;
        }
        for (auto& pk : parked_cars) {
            pk.is_visible_to_ego = has_360_line_of_sight(pk.pos);
        }
        for (auto& sp : parking_spaces) {
            sp.is_visible_to_ego = has_360_line_of_sight(sp.center);
        }
        for (auto& bike : bicycles) {
            bike.is_visible_to_ego = has_360_line_of_sight(bike.pos);
        }
        for (auto& ped : pedestrians) {
            ped.is_visible_to_ego = has_360_line_of_sight(ped.pos);
        }
    }

    void randomize_and_reset_city(uint32_t new_seed = 0) {
        if (new_seed == 0) {
            auto now_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            city_seed = rng.next_u32() ^ static_cast<uint32_t>(now_ns) ^ 0x85EBCA6Bu;
        } else {
            city_seed = new_seed;
        }
        rng = FastRNG(city_seed);
        build_city_network();
        spawn_entities();
    }

    void reset_simulation() {
        randomize_and_reset_city(0);
    }

    void build_city_network() {
        intersections.clear();
        edges.clear();
        buildings.clear();
        bus_stops.clear();
        parking_lots.clear();
        parking_spaces.clear();
        parked_cars.clear();

        // 1. Create 3x3 intersections
        for (int gy = 0; gy < GRID_N; ++gy) {
            for (int gx = 0; gx < GRID_N; ++gx) {
                Intersection inter;
                inter.id = node_index(gx, gy);
                inter.grid_x = gx;
                inter.grid_y = gy;
                inter.center = {
                    (gx - 1) * BLOCK_SPACING,
                    (gy - 1) * BLOCK_SPACING
                };
                inter.half_size = INTER_HALF;

                int roll = rng.range_int(0, 99);
                if (roll < 35)      inter.ctrl_type = IntersectionControlType::TRAFFIC_LIGHTS;
                else if (roll < 58) inter.ctrl_type = IntersectionControlType::POLICE_OFFICER;
                else if (roll < 80) inter.ctrl_type = IntersectionControlType::FOUR_WAY_STOP;
                else                inter.ctrl_type = IntersectionControlType::TWO_WAY_YIELD;

                inter.major_is_ew = (rng.range_int(0, 1) == 0);
                inter.phase = (rng.range_int(0, 1) == 0) ? 0 : 3;
                inter.green_duration = rng.range(9.5f, 13.0f);
                inter.yellow_duration = 3.2f;
                inter.all_red_duration = 1.2f;
                inter.phase_timer = rng.range(0.0f, inter.green_duration * 0.7f);
                intersections.push_back(inter);
            }
        }

        // Center Node 4 is a Grand Multi-Branch 2-Lane Roundabout Plaza with 6 to 8 converging branches
        intersections[4].shape = IntersectionShape::ROUNDABOUT_PLAZA;
        intersections[4].half_size = 11.2f;
        intersections[4].ctrl_type = (rng.range_int(0, 1) == 0) ? IntersectionControlType::TWO_WAY_YIELD
                                                                : IntersectionControlType::POLICE_OFFICER;
        intersections[1].ctrl_type = IntersectionControlType::TRAFFIC_LIGHTS;
        intersections[3].ctrl_type = IntersectionControlType::FOUR_WAY_STOP;
        intersections[5].ctrl_type = IntersectionControlType::POLICE_OFFICER;
        intersections[7].ctrl_type = IntersectionControlType::TRAFFIC_LIGHTS;

        auto populate_edge_samples = [](LaneEdge& e, int num_steps) {
            e.samples.clear();
            float cum_s = 0.0f;
            Vec2 prev_p = e.curve.eval(0.0f);
            for (int i = 0; i <= num_steps; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(num_steps);
                Vec2 p = e.curve.eval(t);
                if (i > 0) cum_s += (p - prev_p).len();
                prev_p = p;

                Vec2 tan = e.curve.tangent(t);
                PathSample ps;
                ps.pos = p;
                ps.heading = std::atan2(tan.y, tan.x);
                ps.curvature = e.curve.curvature(t);
                ps.s = cum_s;
                ps.edge_id = e.id;
                e.samples.push_back(ps);
            }
            e.length = cum_s;
        };

        // Populate a 3-stage counter-clockwise circular roundabout trajectory that strictly stays inside
        // Roundabout Lane 1 (Outer Lane, r = 7.75m) or Roundabout Lane 2 (Inner Lane, r = 5.15m)
        // and NEVER crosses the central grass island (r_island = 3.45m)
        auto populate_roundabout_samples = [](LaneEdge& conn, Vec2 center, Vec2 p0, float in_ang,
                                              Vec2 p3, float out_ang, float r_ring) {
            float th0 = std::atan2(p0.y - center.y, p0.x - center.x);
            float th3 = std::atan2(p3.y - center.y, p3.x - center.x);
            float ccw_sweep = th3 - th0;
            while (ccw_sweep <= 0.12f) ccw_sweep += TWO_PI;

            float blend_ang = std::min(0.40f, ccw_sweep * 0.26f);
            float th_enter = th0 + blend_ang;
            float th_exit  = th0 + ccw_sweep - blend_ang;

            Vec2 p_enter = center + Vec2{std::cos(th_enter), std::sin(th_enter)} * r_ring;
            Vec2 t_enter = Vec2{-std::sin(th_enter), std::cos(th_enter)};
            Vec2 p_exit  = center + Vec2{std::cos(th_exit),  std::sin(th_exit)}  * r_ring;
            Vec2 t_exit  = Vec2{-std::sin(th_exit),  std::cos(th_exit)};

            float d_en = (p_enter - p0).len() * 0.36f;
            CubicBezier2D bez_in = {p0, p0 + Vec2::from_angle(in_ang) * d_en, p_enter - t_enter * d_en, p_enter};

            float d_ex = (p3 - p_exit).len() * 0.36f;
            CubicBezier2D bez_out = {p_exit, p_exit + t_exit * d_ex, p3 - Vec2::from_angle(out_ang) * d_ex, p3};

            std::vector<Vec2> raw_pts;
            constexpr int N_EN = 10, N_ARC = 24, N_EX = 10;
            for (int k = 0; k < N_EN; ++k) {
                float u = static_cast<float>(k) / N_EN;
                Vec2 p = bez_in.eval(u);
                Vec2 rel = p - center;
                if (rel.len() < r_ring) p = center + rel.normalized() * r_ring;
                raw_pts.push_back(p);
            }
            for (int k = 0; k <= N_ARC; ++k) {
                float u = static_cast<float>(k) / N_ARC;
                float th = th_enter + (th_exit - th_enter) * u;
                raw_pts.push_back(center + Vec2{std::cos(th), std::sin(th)} * r_ring);
            }
            for (int k = 1; k <= N_EX; ++k) {
                float u = static_cast<float>(k) / N_EX;
                Vec2 p = bez_out.eval(u);
                Vec2 rel = p - center;
                if (rel.len() < r_ring) p = center + rel.normalized() * r_ring;
                raw_pts.push_back(p);
            }

            conn.samples.clear();
            float cum_s = 0.0f;
            for (size_t k = 0; k < raw_pts.size(); ++k) {
                if (k > 0) cum_s += (raw_pts[k] - raw_pts[k - 1]).len();
                Vec2 tan;
                if (k == 0) tan = raw_pts[1] - raw_pts[0];
                else if (k + 1 == raw_pts.size()) tan = raw_pts[k] - raw_pts[k - 1];
                else tan = raw_pts[k + 1] - raw_pts[k - 1];

                PathSample ps;
                ps.pos = raw_pts[k];
                ps.heading = std::atan2(tan.y, tan.x);
                ps.curvature = 1.0f / r_ring;
                ps.s = cum_s;
                ps.edge_id = conn.id;
                conn.samples.push_back(ps);
            }
            conn.samples.front().heading = in_ang;
            conn.samples.back().heading = out_ang;
            conn.length = cum_s;
            conn.curve = {p0, p_enter, p_exit, p3};
        };

        // Helper to add a directed straight road edge between node u and node v
        auto add_road_edge = [&](int u, int v, bool is_diag, bool one_way, int lanes_same_dir) -> int {
            Vec2 cu = intersections[u].center;
            Vec2 cv = intersections[v].center;
            Vec2 d = (cv - cu).normalized();
            Vec2 right = d.perp_right();
            float ang = std::atan2(d.y, d.x);

            // Map exact angle to nearest cardinal signal group (0=E, 1=N, 2=W, 3=S)
            int card_dir = 0;
            if (std::fabs(d.x) >= std::fabs(d.y)) card_dir = (d.x >= 0.0f) ? 0 : 2;
            else                                  card_dir = (d.y >= 0.0f) ? 1 : 3;

            float hu = intersections[u].half_size;
            float hv = intersections[v].half_size;
            // Always place the primary driving lane in Lane 1 (the rightmost curb lane):
            //   - On 4-lane boulevards (!one_way && lanes_same_dir >= 2), Lane 1 is at +3.95m (outside the 2.65m dashed divider)
            //   - On 2-lane one-way streets and standard 2-way roads, Lane 1 is at +1.95m (outside the 0.0m center divider)
            float lat_off = (!one_way && lanes_same_dir >= 2) ? 3.95f : 1.95f;

            Vec2 p_start = cu + d * hu + right * lat_off;
            Vec2 p_end   = cv - d * hv + right * lat_off;

            LaneEdge edge;
            edge.id = static_cast<int>(edges.size());
            edge.is_intersection_connector = false;
            edge.from_node = u;
            edge.to_node = v;
            edge.entry_dir = card_dir;
            edge.exit_dir = card_dir;
            edge.exact_angle = ang;
            edge.is_diagonal = is_diag;
            edge.is_one_way = one_way;
            edge.num_lanes_same_dir = lanes_same_dir;
            edge.turn = TurnIntent::STRAIGHT;
            edge.curve = {
                p_start,
                p_start * 0.67f + p_end * 0.33f,
                p_start * 0.33f + p_end * 0.67f,
                p_end
            };
            populate_edge_samples(edge, 56);
            edges.push_back(edge);

            // Record branch angles on endpoints
            auto add_branch_ang = [&](int node_id, float a) {
                for (float existing : intersections[node_id].branch_angles) {
                    if (std::fabs(wrap_angle(existing - a)) < 0.25f) return;
                }
                intersections[node_id].branch_angles.push_back(a);
            };
            add_branch_ang(u, ang);
            add_branch_ang(v, wrap_angle(ang + PI));
            return edge.id;
        };

        // 2. Create Orthogonal Grid Roads:
        //    - Central avenues (connecting to node 4) are Multi-Lane 4-Lane Boulevards (2 lanes each direction)
        //    - Randomly 1 perimeter avenue is a 3-Branch T-Junction (omitted link)
        //    - Randomly 2 perimeter links are 2-Lane ONE-WAY Streets
        int omitted_pair = rng.range_int(0, 3);
        int one_way_pair = (omitted_pair + 1 + rng.range_int(0, 2)) % 4;

        auto get_perim_pair = [](int idx, int& u, int& v) {
            if (idx == 0)      { u = 0; v = 1; }
            else if (idx == 1) { u = 1; v = 2; }
            else if (idx == 2) { u = 6; v = 7; }
            else               { u = 7; v = 8; }
        };
        int om_u = -1, om_v = -1, ow_u = -1, ow_v = -1;
        get_perim_pair(omitted_pair, om_u, om_v);
        get_perim_pair(one_way_pair, ow_u, ow_v);

        for (int gy = 0; gy < GRID_N; ++gy) {
            for (int gx = 0; gx < GRID_N; ++gx) {
                int u = node_index(gx, gy);
                for (int dir = 0; dir < 2; ++dir) { // 0=East(+X), 1=North(+Y) undirected pairs
                    int nx = gx + (dir == 0 ? 1 : 0);
                    int ny = gy + (dir == 1 ? 1 : 0);
                    if (nx >= GRID_N || ny >= GRID_N) continue;
                    int v = node_index(nx, ny);

                    if ((u == om_u && v == om_v) || (u == om_v && v == om_u)) {
                        continue; // Omitted road creates a genuine 3-way T-Junction
                    }

                    bool is_central_boulevard = (u == 4 || v == 4);
                    bool is_one_way_link = ((u == ow_u && v == ow_v) || (u == ow_v && v == ow_u));

                    if (is_one_way_link) {
                        // 2-Lane One-Way Street. (Ensure direction flows smoothly around perimeter)
                        bool forward_dir = (rng.range_int(0, 1) == 0);
                        add_road_edge(forward_dir ? u : v, forward_dir ? v : u, false, true, 2);
                    } else {
                        int lanes = is_central_boulevard ? 2 : 1;
                        int e1 = add_road_edge(u, v, false, false, lanes);
                        int e2 = add_road_edge(v, u, false, false, lanes);
                        edges[e1].oppose_edge_id = e2;
                        edges[e2].oppose_edge_id = e1;
                    }
                }
            }
        }

        // 3. Add Non-Crossing Quadrant Diagonal Avenues:
        //    Each of the 4 quadrants chooses EITHER:
        //      (A) A Radial Boulevard into Center Node 4 (0<->4, 2<->4, 6<->4, 8<->4) -> creates 6/7-Way Grand Roundabout
        //      (B) A Diamond Chord Avenue between Midpoint Nodes (1<->3, 1<->5, 3<->7, 5<->7) -> creates 4/5-Way Skewed Hubs
        struct QuadDiag {
            int radial_corner;
            int chord_u;
            int chord_v;
        };
        QuadDiag quads[4] = {
            {0, 1, 3}, // SW quadrant
            {2, 1, 5}, // SE quadrant
            {6, 3, 7}, // NW quadrant
            {8, 5, 7}  // NE quadrant
        };
        int start_q = rng.range_int(0, 3);
        for (int q = 0; q < 4; ++q) {
            int qi = (start_q + q) % 4;
            bool use_radial = (q < 2); // 2 radial avenues into Node 4 (6-way center) + 2 diamond chords on midpoints (5-way / 4-way hubs)
            int u = use_radial ? quads[qi].radial_corner : quads[qi].chord_u;
            int v = use_radial ? 4                       : quads[qi].chord_v;
            bool diag_one_way = (q == 3); // Make one diamond chord a 2-lane One-Way Avenue
            if (diag_one_way) {
                add_road_edge(u, v, true, true, 2);
            } else {
                int e1 = add_road_edge(u, v, true, false, 1);
                int e2 = add_road_edge(v, u, true, false, 1);
                edges[e1].oppose_edge_id = e2;
                edges[e2].oppose_edge_id = e1;
            }
        }

        // Update intersection shapes based on their true connected branch count
        // Note: only Node 4 is a 2-Lane Circular Roundabout Plaza (half_size = 11.2m), while 5-branch perimeter nodes
        // are Multi-Branch Star Hubs so all road endpoints match their exact half_size
        for (auto& inter : intersections) {
            inter.branch_count = static_cast<int>(inter.branch_angles.size());
            if (inter.id == 4) {
                inter.shape = IntersectionShape::ROUNDABOUT_PLAZA;
            } else if (inter.branch_count >= 5) {
                inter.shape = IntersectionShape::MULTI_BRANCH_STAR;
            } else if (inter.branch_count == 4) {
                inter.shape = IntersectionShape::CROSS_PLUS;
            } else {
                inter.shape = IntersectionShape::T_JUNCTION;
            }
        }

        int num_straight_edges = static_cast<int>(edges.size());

        // 4. Create Intersection Turn Connectors across all converging branches (3-way, 4-way, 5-way, 6-way, 7-way)
        for (int i = 0; i < num_straight_edges; ++i) {
            int mid = edges[i].to_node;
            float in_ang = edges[i].exact_angle;
            Vec2 p0 = edges[i].samples.back().pos;
            Vec2 d_in = Vec2::from_angle(in_ang);
            bool is_roundabout = (intersections[mid].shape == IntersectionShape::ROUNDABOUT_PLAZA);

            int non_hairpin_exits = 0;
            for (int j = 0; j < num_straight_edges; ++j) {
                if (edges[j].from_node == mid && edges[j].to_node != edges[i].from_node) {
                    float da = std::fabs(wrap_angle(edges[j].exact_angle - in_ang));
                    if (is_roundabout || da < 2.15f) {
                        ++non_hairpin_exits;
                    }
                }
            }

            for (int j = 0; j < num_straight_edges; ++j) {
                if (edges[j].from_node != mid) continue;
                float out_ang = edges[j].exact_angle;
                float delta_ang = wrap_angle(out_ang - in_ang);
                if (!is_roundabout && non_hairpin_exits > 0) {
                    if (edges[j].to_node == edges[i].from_node || std::fabs(delta_ang) >= 2.15f) {
                        continue; // Avoid 135-deg/180-deg hairpin U-turns on single-lane intersections when normal exits exist
                    }
                } else if (is_roundabout && edges[j].to_node == edges[i].from_node && non_hairpin_exits > 0) {
                    continue;
                }

                Vec2 p3 = edges[j].samples.front().pos;
                Vec2 d_out = Vec2::from_angle(out_ang);

                TurnIntent turn_type = TurnIntent::STRAIGHT;
                if (delta_ang > 0.35f)       turn_type = TurnIntent::LEFT;
                else if (delta_ang < -0.35f) turn_type = TurnIntent::RIGHT;
                else                         turn_type = TurnIntent::STRAIGHT;

                LaneEdge conn;
                conn.id = static_cast<int>(edges.size());
                conn.is_intersection_connector = true;
                conn.from_node = mid;
                conn.to_node = mid;
                conn.entry_dir = edges[i].exit_dir;
                conn.exit_dir = edges[j].entry_dir;
                conn.exact_angle = out_ang;
                conn.turn = turn_type;

                if (is_roundabout) {
                    // Use Outer Roundabout Lane 1 (r = 7.75m) for Right & Straight exits,
                    // and Inner Roundabout Lane 2 (r = 5.15m) for Left turns (>180 deg circulation)
                    float r_ring = (turn_type == TurnIntent::LEFT) ? 5.15f : 7.75f;
                    populate_roundabout_samples(conn, intersections[mid].center, p0, in_ang, p3, out_ang, r_ring);
                } else {
                    float chord = (p3 - p0).len();
                    float ctrl_dist = clampf(chord * 0.42f, 2.8f, 6.5f);
                    Vec2 c1 = p0 + d_in * ctrl_dist;
                    Vec2 c2 = p3 - d_out * ctrl_dist;
                    conn.curve = {p0, c1, c2, p3};
                    populate_edge_samples(conn, 24);
                }

                conn.next_edges.push_back(j);
                edges[i].next_edges.push_back(conn.id);
                edges.push_back(conn);
            }
        }

        // 5. Place Designated "BUS" Pavement Stop Zones & Shelters strictly inside Lane 1 (Outermost Right Curb Lane)
        int bus_stop_id = 0;
        for (int i = 0; i < num_straight_edges && bus_stop_id < 4; i += std::max(1, num_straight_edges / 4)) {
            const auto& e = edges[i];
            if (e.length < 40.0f) continue;

            BusStopZone bsz;
            bsz.id = ++bus_stop_id;
            bsz.edge_id = e.id;
            bsz.s_center = e.length * 0.50f;
            bsz.length = 13.5f;
            float t = 0.50f;
            Vec2 p_center = e.curve.eval(t);
            Vec2 tan = e.curve.tangent(t).normalized();
            bsz.heading = std::atan2(tan.y, tan.x);
            bsz.pos = p_center + tan.perp_right() * 0.12f;
            bus_stops.push_back(bsz);
        }

        Color3 parked_palette[6] = {
            {0.44f, 0.48f, 0.54f},
            {0.78f, 0.82f, 0.86f},
            {0.25f, 0.34f, 0.48f},
            {0.62f, 0.32f, 0.22f},
            {0.20f, 0.48f, 0.38f},
            {0.58f, 0.52f, 0.32f}
        };
        int pk_id = 0;
        int space_id = 0;

        auto edge_has_bus_or_parking = [&](int eid, int opp_id) -> bool {
            for (const auto& bs : bus_stops) {
                if (bs.edge_id == eid || (opp_id >= 0 && bs.edge_id == opp_id)) return true;
            }
            for (const auto& lot : parking_lots) {
                if (lot.road_edge_id == eid || (opp_id >= 0 && lot.road_edge_id == opp_id)) return true;
            }
            for (const auto& sp : parking_spaces) {
                if (sp.road_edge_id == eid || (opp_id >= 0 && sp.road_edge_id == opp_id)) return true;
            }
            for (const auto& pk : parked_cars) {
                if (pk.edge_id == eid || (opp_id >= 0 && pk.edge_id == opp_id)) return true;
            }
            return false;
        };

        // 6A. Create 3 Rectangular Off-Street Parking Lot Zones alongside Roads (with 3 Distinct Layouts):
        //     - Lot 1: PERPENDICULAR_90_REVERSE (90-deg perpendicular stalls requiring Reverse-In parking)
        //     - Lot 2: PERPENDICULAR_90_FRONT   (90-deg perpendicular stalls with Front-In parking)
        //     - Lot 3: ANGLED_45_FRONT          (45-deg angled echelon stalls with Front-In parking)
        ParkingLayoutType lot_layouts[3] = {
            ParkingLayoutType::PERPENDICULAR_90_REVERSE,
            ParkingLayoutType::PERPENDICULAR_90_FRONT,
            ParkingLayoutType::ANGLED_45_FRONT
        };
        int lot_count = 0;
        for (int i = 0; i < num_straight_edges && lot_count < 3; ++i) {
            const auto& e = edges[i];
            if (e.length < 48.0f || e.is_diagonal) continue;
            if (edge_has_bus_or_parking(e.id, e.oppose_edge_id)) continue;

            float s_mid = e.length * 0.50f;
            Vec2 p_road = e.curve.eval(0.50f);
            Vec2 d = e.curve.tangent(0.50f).normalized();
            Vec2 right = d.perp_right();
            float road_h = std::atan2(d.y, d.x);

            ParkingLotZone lot;
            lot.id = ++lot_count;
            lot.road_edge_id = e.id;
            lot.heading = road_h;
            lot.length = 26.0f;
            lot.width = 12.6f;
            lot.center = p_road + right * 7.85f;
            lot.driveway_in_pt = p_road - d * 12.5f + right * 1.4f;
            lot.driveway_out_pt = p_road + d * 12.2f + right * 0.4f;
            lot.exit_edge_s = std::min(e.length - 6.0f, s_mid + 12.4f);
            lot.layout = lot_layouts[(lot.id - 1) % 3];
            parking_lots.push_back(lot);

            Vec2 aisle_center = p_road + right * 4.50f;
            Vec2 row_center   = p_road + right * 10.35f;

            // Create 5 contiguous rectangular parking stalls inside this Rectangular Parking Lot Zone
            for (int k = 0; k < 5; ++k) {
                float u_stall = (k - 2) * 3.40f; // -6.8m, -3.4m, 0.0m, +3.4m, +6.8m
                ParkingSpace sp;
                sp.id = ++space_id;
                sp.road_edge_id = e.id;
                sp.lot_id = lot.id;
                sp.is_on_lane_one = false;
                sp.layout = lot.layout;
                sp.center = row_center + d * u_stall;
                sp.length = 5.20f;
                sp.width = 2.90f;
                sp.road_s = s_mid + u_stall;
                // Waypoint 0 is ON the road at the upstream driveway opening (u = -12.6m);
                // Waypoint 1 is inside the lot aisle (u = -7.6m) so the car stays on the road until the driveway
                // and has plenty of distance inside the aisle to straighten out before entering any stall
                sp.road_turn_in_pt = p_road - d * 12.6f;
                sp.aisle_entry_pt  = aisle_center - d * 7.6f;
                sp.aisle_heading   = road_h;
                sp.aisle_exit_pt   = aisle_center + d * 7.6f;
                sp.exit_merge_pt   = lot.driveway_out_pt;
                sp.exit_road_s     = lot.exit_edge_s;

                Vec2 p_aisle_k = aisle_center + d * u_stall;
                // Note: In our 2D coordinates, right = d.perp_right() has angle (road_h - PI/2) pointing INWARD into the bay,
                // while -right = d.perp_left() has angle (road_h + PI/2) pointing OUTWARD toward the lot aisle
                if (sp.layout == ParkingLayoutType::PERPENDICULAR_90_REVERSE) {
                    sp.heading = wrap_angle(road_h + PI * 0.5f); // Faces OUTWARD toward aisle (-right) when backed in
                    sp.setup_pt = p_aisle_k + d * 4.35f - right * 0.50f;
                    sp.setup_heading = wrap_angle(road_h + 0.28f);
                    sp.bay_mouth_pt = sp.center - right * 2.40f;
                } else if (sp.layout == ParkingLayoutType::PERPENDICULAR_90_FRONT) {
                    sp.heading = wrap_angle(road_h - PI * 0.5f); // Faces INWARD (+right) into the bay when parked front-first
                    sp.road_turn_in_pt = p_road - d * 13.8f;
                    sp.aisle_entry_pt = aisle_center - d * 9.2f - right * 0.70f;
                    sp.setup_pt = p_aisle_k - d * 4.45f - right * 0.70f;
                    sp.setup_heading = wrap_angle(road_h - 0.30f);
                    sp.bay_mouth_pt = sp.center - right * 2.40f;
                } else {
                    // ANGLED_45_FRONT: 45-degree right-angled echelon stall pointing inward (+right and +d)
                    sp.heading = wrap_angle(road_h - PI * 0.25f);
                    sp.width = 2.80f;
                    Vec2 sfwd = Vec2::from_angle(sp.heading);
                    sp.setup_pt = sp.center - sfwd * 5.75f;
                    sp.setup_heading = wrap_angle(road_h - 0.36f);
                    sp.bay_mouth_pt = sp.center - sfwd * 2.40f;
                }

                // Make 2 of the 5 stalls occupied by stationary parked cars, leaving 3 vacant
                // For front-in lots, occupy k=0,1 so vacant stalls (k=2,3,4 at u >= 0.0m) have a long straight aisle lead-in
                bool occupy = (sp.layout == ParkingLayoutType::PERPENDICULAR_90_REVERSE)
                              ? (k == 0 || k == 3)
                              : (k == 0 || k == 1);
                sp.is_occupied = occupy;
                if (occupy) {
                    HalfParkedCar pk;
                    pk.id = ++pk_id;
                    pk.edge_id = -1; // Inside off-street parking lot, does not block main road
                    pk.parking_space_id = sp.id;
                    pk.s_pos = sp.road_s;
                    pk.pos = sp.center;
                    pk.heading = sp.heading;
                    pk.body_color = parked_palette[(pk_id + k) % 6];
                    pk.hazard_lights_on = false;
                    parked_cars.push_back(pk);
                    sp.occupant_car_id = pk.id;
                }
                parking_spaces.push_back(sp);
            }
        }

        // 6B. Create 4 On-Street Lateral (Parallel) Parking Rows (3 Contiguous Rectangular Stalls per Row):
        //     - 2 rows painted directly ON THE FIRST LANE (Lane 1) of Multi-Lane Boulevards (`is_on_lane_one = true`)
        //     - 2 rows painted in Recessed Curbside Parallel Bays alongside standard roads (`is_on_lane_one = false`)
        int lateral_rows_lane1 = 0;
        int lateral_rows_curb  = 0;
        for (int i = 0; i < num_straight_edges && (lateral_rows_lane1 < 2 || lateral_rows_curb < 2); ++i) {
            const auto& e = edges[i];
            if (e.length < 48.0f) continue;
            if (edge_has_bus_or_parking(e.id, e.oppose_edge_id)) continue;

            bool can_be_lane1 = (e.num_lanes_same_dir >= 2 && lateral_rows_lane1 < 2);
            bool can_be_curb  = (e.num_lanes_same_dir == 1 && lateral_rows_curb < 2);
            if (!can_be_lane1 && !can_be_curb) continue;

            bool on_lane_one = can_be_lane1;
            if (on_lane_one) ++lateral_rows_lane1;
            else             ++lateral_rows_curb;

            float s_mid = e.length * 0.48f;
            Vec2 p_mid = e.curve.eval(0.48f);
            Vec2 d = e.curve.tangent(0.48f).normalized();
            Vec2 right = d.perp_right();
            float road_h = std::atan2(d.y, d.x);

            // Lateral offset of the parallel parking row relative to e.curve (Lane 1 centerline):
            //   - If on_lane_one == true: right on Lane 1 (+0.20m), with Lane 2 (-2.50m left) used for approach & reverse S-curve
            //   - If on_lane_one == false: in a recessed curbside bay (+2.75m right of Lane 1, 100% clear of Lane 1),
            //     with Lane 1 (0.00m) used for approach & reverse S-curve
            float row_lat_right = on_lane_one ? 0.20f : 2.75f;
            float maneuver_lat_right = on_lane_one ? -2.50f : 0.00f;

            for (int k = 0; k < 3; ++k) {
                float u_stall = (k - 1) * 6.80f; // -6.8m (k=0), 0.0m (k=1), +6.8m (k=2)
                ParkingSpace sp;
                sp.id = ++space_id;
                sp.road_edge_id = e.id;
                sp.lot_id = -1;
                sp.is_on_lane_one = on_lane_one;
                sp.layout = ParkingLayoutType::PARALLEL_LATERAL_REVERSE;
                sp.center = p_mid + d * u_stall + right * row_lat_right;
                sp.heading = road_h;
                sp.length = 6.30f;
                sp.width = 2.30f;
                sp.road_s = s_mid + u_stall;

                Vec2 man_base = p_mid + d * u_stall + right * maneuver_lat_right;
                sp.road_turn_in_pt = p_mid + d * (u_stall - 8.5f);
                sp.aisle_entry_pt  = man_base - d * 3.5f;
                sp.aisle_heading   = road_h;
                sp.setup_pt        = man_base + d * 6.00f; // Pull 6.00m ahead of the lateral stall in the setup lane for a kinematically exact reverse S-curve
                sp.setup_heading   = road_h;
                sp.bay_mouth_pt    = sp.center - right * 1.35f + d * 2.40f;
                sp.aisle_exit_pt   = man_base + d * 3.40f;
                sp.exit_merge_pt   = man_base + d * 6.20f;
                sp.exit_road_s     = std::min(e.length - 5.0f, sp.road_s + 6.20f);

                // Make the downstream stall (k == 2 at +6.8m) occupied by a parked car, leaving upstream stalls (k == 0, 1) vacant
                // so approaching from upstream into Lane 2 never clips an occupied car at the row entrance
                bool occupy = (k == 2);
                sp.is_occupied = occupy;
                if (occupy) {
                    HalfParkedCar pk;
                    pk.id = ++pk_id;
                    // If this occupied lateral stall is on Lane 1 of a multi-lane road, normal traffic must overtake it
                    pk.edge_id = on_lane_one ? e.id : -1;
                    pk.parking_space_id = sp.id;
                    pk.s_pos = sp.road_s;
                    pk.pos = sp.center;
                    pk.heading = sp.heading;
                    pk.body_color = parked_palette[(pk_id + k) % 6];
                    pk.hazard_lights_on = false;
                    parked_cars.push_back(pk);
                    sp.occupant_car_id = pk.id;
                }
                parking_spaces.push_back(sp);
            }
        }

        // 7. Spawn 2 Additional Half-Lane Hazard Cars on Open Overtaking Stretches (Away from Bus Stops & Parking Zones)
        int hazard_cars = 0;
        for (int i = 0; i < num_straight_edges && hazard_cars < 2; ++i) {
            const auto& e = edges[i];
            if (e.length < 46.0f) continue;
            if (edge_has_bus_or_parking(e.id, e.oppose_edge_id)) continue;

            HalfParkedCar pk;
            pk.id = ++pk_id;
            pk.edge_id = e.id;
            pk.parking_space_id = -1;
            float frac = 0.50f;
            pk.s_pos = e.length * frac;
            Vec2 p_center = e.curve.eval(frac);
            Vec2 tan = e.curve.tangent(frac).normalized();
            pk.heading = std::atan2(tan.y, tan.x);
            pk.pos = p_center + tan.perp_right() * 1.25f;
            pk.body_color = parked_palette[pk_id % 6];
            pk.hazard_lights_on = true;
            parked_cars.push_back(pk);
            ++hazard_cars;
        }

        // 8. Procedurally generate 3D City Buildings (filtering out any building that overlaps any road,
        //    Rectangular Parking Lot Zone, or Curbside Lateral Parking Bay)
        auto does_building_overlap_any_road_or_parking = [&](Vec2 bmin, Vec2 bmax) -> bool {
            Vec2 pad_min = {bmin.x - 6.8f, bmin.y - 6.8f};
            Vec2 pad_max = {bmax.x + 6.8f, bmax.y + 6.8f};
            for (int i = 0; i < num_straight_edges; ++i) {
                const auto& e = edges[i];
                if (segment_intersects_aabb(e.samples.front().pos, e.samples.back().pos, pad_min, pad_max)) {
                    return true;
                }
            }
            for (const auto& lot : parking_lots) {
                Vec2 d = Vec2::from_angle(lot.heading);
                Vec2 r = d.perp_right();
                float hl = lot.length * 0.5f + 3.2f;
                float hw = lot.width * 0.5f + 3.2f;
                Vec2 corners[4] = {
                    lot.center - d * hl - r * hw,
                    lot.center + d * hl - r * hw,
                    lot.center + d * hl + r * hw,
                    lot.center - d * hl + r * hw
                };
                for (int c = 0; c < 4; ++c) {
                    if (segment_intersects_aabb(corners[c], corners[(c + 1) & 3],
                                                {bmin.x - 2.0f, bmin.y - 2.0f}, {bmax.x + 2.0f, bmax.y + 2.0f})) {
                        return true;
                    }
                }
                if (lot.center.x >= bmin.x - hl && lot.center.x <= bmax.x + hl &&
                    lot.center.y >= bmin.y - hl && lot.center.y <= bmax.y + hl) {
                    return true;
                }
            }
            for (const auto& sp : parking_spaces) {
                if (sp.center.x >= bmin.x - 4.5f && sp.center.x <= bmax.x + 4.5f &&
                    sp.center.y >= bmin.y - 4.5f && sp.center.y <= bmax.y + 4.5f) {
                    return true;
                }
            }
            return false;
        };

        auto add_block_buildings = [&](float cx, float cy, float half_w) {
            float pad = 3.5f;
            float sub_base = (half_w - pad) * 0.44f;
            float offsets[2] = {-half_w * 0.48f, half_w * 0.48f};
            for (int by = 0; by < 2; ++by) {
                for (int bx = 0; bx < 2; ++bx) {
                    if (rng.range_int(0, 99) < 16) continue;

                    float sx = sub_base * rng.range(0.70f, 0.96f);
                    float sy = sub_base * rng.range(0.70f, 0.96f);
                    Vec2 bc = {cx + offsets[bx], cy + offsets[by]};
                    Vec2 bmin = {bc.x - sx, bc.y - sy};
                    Vec2 bmax = {bc.x + sx, bc.y + sy};

                    if (does_building_overlap_any_road_or_parking(bmin, bmax)) continue;

                    Building3D b;
                    b.min_pt = bmin;
                    b.max_pt = bmax;
                    b.height = rng.range(7.5f, 25.0f);

                    float r_tint = rng.range(0.20f, 0.38f);
                    float g_tint = r_tint + rng.range(-0.02f, 0.05f);
                    float b_tint = g_tint + rng.range(0.02f, 0.10f);
                    b.color = {r_tint, g_tint, b_tint};
                    b.roof_color = b.color * 0.82f;
                    buildings.push_back(b);
                }
            }
        };

        float block_half = (BLOCK_SPACING - INTER_HALF * 2.0f) * 0.5f - 2.5f;
        for (int gy = -1; gy <= GRID_N - 1; ++gy) {
            for (int gx = -1; gx <= GRID_N - 1; ++gx) {
                float cx = (gx - 0.5f) * BLOCK_SPACING;
                float cy = (gy - 0.5f) * BLOCK_SPACING;
                add_block_buildings(cx, cy, block_half);
            }
        }
    }

    void ensure_route_horizon(Vehicle& v) {
        if (v.route_edges.empty()) {
            v.route_edges.push_back(0);
            v.current_route_idx = 0;
        }
        while (static_cast<int>(v.route_edges.size()) - v.current_route_idx < 6) {
            int last_edge_id = v.route_edges.back();
            const LaneEdge& last_edge = edges[last_edge_id];
            if (last_edge.next_edges.empty()) break;

            int n_choices = static_cast<int>(last_edge.next_edges.size());
            int pick = last_edge.next_edges[rng.range_int(0, n_choices - 1)];

            if (n_choices > 1) {
                for (int attempt = 0; attempt < 4; ++attempt) {
                    int candidate = last_edge.next_edges[rng.range_int(0, n_choices - 1)];
                    int target_after = edges[candidate].next_edges.empty() ? candidate : edges[candidate].next_edges[0];
                    bool recently_seen = false;
                    int hist_start = std::max(0, static_cast<int>(v.route_edges.size()) - 6);
                    for (int h = hist_start; h < static_cast<int>(v.route_edges.size()); ++h) {
                        if (v.route_edges[h] == candidate || v.route_edges[h] == target_after) {
                            recently_seen = true;
                            break;
                        }
                    }
                    if (!recently_seen) {
                        pick = candidate;
                        break;
                    }
                }
            }
            v.route_edges.push_back(pick);
        }

        v.active_intent = TurnIntent::STRAIGHT;
        for (int k = v.current_route_idx; k < std::min(static_cast<int>(v.route_edges.size()), v.current_route_idx + 2); ++k) {
            const LaneEdge& e = edges[v.route_edges[k]];
            if (e.is_intersection_connector) {
                v.active_intent = e.turn;
                break;
            }
        }
    }

    void ensure_bicycle_route(Bicycle& b) {
        if (b.route_edges.empty()) {
            b.route_edges.push_back(0);
            b.current_route_idx = 0;
        }
        while (static_cast<int>(b.route_edges.size()) - b.current_route_idx < 5) {
            int last_id = b.route_edges.back();
            const LaneEdge& le = edges[last_id];
            if (le.next_edges.empty()) break;
            int chosen = le.next_edges[rng.range_int(0, static_cast<int>(le.next_edges.size()) - 1)];
            b.route_edges.push_back(chosen);
        }
    }

    void update_vehicle_route_progress(Vehicle& v, float* out_cte = nullptr, float* out_heading_err = nullptr) {
        ensure_route_horizon(v);

        const LaneEdge& cur_e = edges[v.route_edges[v.current_route_idx]];
        bool can_advance_edge = (v.current_edge_s >= cur_e.length - 0.65f);

        float best_dist_sq = 1e9f;
        int best_route_slot = v.current_route_idx;
        PathSample best_sample{};

        int search_end = std::min(static_cast<int>(v.route_edges.size()),
                                  v.current_route_idx + (can_advance_edge ? 3 : 1));
        for (int slot = v.current_route_idx; slot < search_end; ++slot) {
            const LaneEdge& e = edges[v.route_edges[slot]];
            for (const auto& ps : e.samples) {
                float d2 = (ps.pos - v.pos).len_sq();
                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    best_route_slot = slot;
                    best_sample = ps;
                }
            }
        }

        if (best_route_slot > v.current_route_idx) {
            v.current_route_idx = best_route_slot;
            v.intersection_wait_time = 0.0f;
            v.stop_sign_hold_timer = 0.0f;
            v.stop_sign_completed_for_edge = false;
            v.last_serviced_bus_stop_id = -1;
            ensure_route_horizon(v);
        }
        v.current_edge_s = best_sample.s;

        Vec2 path_dir = Vec2::from_angle(best_sample.heading);
        Vec2 offset = v.pos - best_sample.pos;
        float cte = path_dir.cross(offset);
        float h_err = wrap_angle(v.heading - best_sample.heading);

        if (out_cte) *out_cte = cte;
        if (out_heading_err) *out_heading_err = h_err;
    }

    void get_route_lookahead(const Vehicle& v, float horizon_m, std::vector<PathSample>& out_samples) const {
        out_samples.clear();
        if (v.route_edges.empty()) return;

        float cum_dist = 0.0f;
        Vec2 prev_pos = v.pos;
        bool started = false;

        for (int slot = v.current_route_idx; slot < static_cast<int>(v.route_edges.size()); ++slot) {
            const LaneEdge& e = edges[v.route_edges[slot]];
            for (const auto& ps : e.samples) {
                if (slot == v.current_route_idx && ps.s + 0.2f < v.current_edge_s) continue;
                if (started) {
                    cum_dist += (ps.pos - prev_pos).len();
                } else {
                    started = true;
                }
                prev_pos = ps.pos;

                PathSample sample = ps;
                sample.s = cum_dist;
                out_samples.push_back(sample);
                if (cum_dist >= horizon_m) return;
            }
        }
    }

    void spawn_entities() {
        sim_time = 0.0f;
        spontaneous_runner_cooldown = 2.5f;
        spontaneous_accident_cooldown = 6.0f;
        active_accident_count = 0;

        std::vector<int> straight_edges;
        for (const auto& e : edges) {
            if (!e.is_intersection_connector) {
                straight_edges.push_back(e.id);
            }
        }

        int ego_pick_idx = rng.range_int(0, static_cast<int>(straight_edges.size()) - 1);
        int start_edge_id = straight_edges[ego_pick_idx];

        ego = Vehicle{};
        ego.id = 0;
        ego.is_ego = true;
        ego.body_color = {0.92f, 0.18f, 0.22f};
        ego.route_edges = {start_edge_id};
        ego.current_route_idx = 0;
        const auto& start_sample = edges[start_edge_id].samples[5];
        ego.pos = start_sample.pos;
        ego.heading = start_sample.heading;
        ego.speed = 6.5f;
        ego.current_edge_s = start_sample.s;
        ensure_route_horizon(ego);

        // Spawn 8 NPC Vehicles + 2 City Transit Buses (Total 10 Traffic Vehicles)
        npc_cars.clear();
        Color3 npc_palette[6] = {
            {0.20f, 0.58f, 0.92f},
            {0.85f, 0.88f, 0.92f},
            {0.25f, 0.80f, 0.55f},
            {0.92f, 0.65f, 0.15f},
            {0.55f, 0.45f, 0.88f},
            {0.35f, 0.40f, 0.48f}
        };

        int num_npcs = 10;
        int spawned = 0;
        for (size_t k = 0; k < straight_edges.size() && spawned < num_npcs; ++k) {
            int eid = straight_edges[(ego_pick_idx + 1 + k * 2) % straight_edges.size()];
            if (eid == start_edge_id) continue;
            const auto& e = edges[eid];
            const auto& ps = e.samples[rng.range_int(8, static_cast<int>(e.samples.size()) - 14)];

            Vehicle npc;
            npc.id = ++spawned;
            npc.is_ego = false;
            npc.is_bus = (spawned <= 2); // First 2 NPCs are City Transit Buses
            if (npc.is_bus) {
                npc.length = 9.60f;
                npc.width  = 2.46f;
                npc.wheelbase = 5.40f;
                npc.body_color = (spawned == 1) ? Color3{0.12f, 0.48f, 0.88f} : Color3{0.16f, 0.68f, 0.52f};
            } else {
                npc.body_color = npc_palette[spawned % 6];
            }
            npc.pos = ps.pos;
            npc.heading = ps.heading;
            npc.speed = npc.is_bus ? 6.0f : rng.range(5.5f, 8.0f);
            npc.route_edges = {eid};
            npc.current_route_idx = 0;
            npc.current_edge_s = ps.s;
            ensure_route_horizon(npc);
            npc_cars.push_back(npc);
        }

        bicycles.clear();
        Color3 bike_jerseys[4] = {
            {0.98f, 0.85f, 0.12f},
            {0.15f, 0.92f, 0.65f},
            {0.98f, 0.42f, 0.15f},
            {0.28f, 0.72f, 0.98f}
        };
        for (int b_i = 0; b_i < 6; ++b_i) {
            int eid = straight_edges[(ego_pick_idx + 1 + b_i * 3) % straight_edges.size()];
            const auto& e = edges[eid];
            size_t s_idx = static_cast<size_t>(rng.range_int(6, static_cast<int>(e.samples.size()) - 10));
            const auto& ps = e.samples[s_idx];

            Bicycle bike;
            bike.id = b_i + 1;
            bike.route_edges = {eid};
            bike.current_route_idx = 0;
            bike.current_edge_s = ps.s;
            bike.heading = ps.heading;
            bike.cruise_speed = rng.range(3.8f, 5.2f);
            bike.speed = bike.cruise_speed;
            bike.lateral_offset = 1.10f;
            Vec2 right_d = Vec2::from_angle(ps.heading).perp_right();
            bike.pos = ps.pos + right_d * bike.lateral_offset;
            bike.rider_color = bike_jerseys[b_i % 4];
            ensure_bicycle_route(bike);
            bicycles.push_back(bike);
        }

        pedestrians.clear();
        int ped_id = 0;
        for (const auto& inter : intersections) {
            for (int arm = 0; arm < 4; ++arm) {
                float arm_hw = 4.15f, cw_dist = inter.half_size + 1.6f;
                if (!get_arm_road_info(inter.id, arm, &arm_hw, &cw_dist)) continue;
                if ((arm + inter.id) % 2 != 0) continue;
                Pedestrian ped;
                ped.id = ++ped_id;
                ped.intersection_id = inter.id;
                ped.crosswalk_dir = arm;
                Vec2 arm_d = dir_vec(arm);
                Vec2 cross_d = arm_d.perp_left();
                Vec2 cw_center = inter.center + arm_d * cw_dist;
                // Place waiting pedestrians on the raised concrete sidewalk outside the actual road half-width (arm_hw + 0.85m)
                ped.start_pt = cw_center - cross_d * (arm_hw + 0.85f);
                ped.end_pt   = cw_center + cross_d * (arm_hw + 0.85f);

                bool walk_ok = inter.is_pedestrian_walk_green(arm);
                ped.progress = (walk_ok && (ped_id % 3 == 0)) ? rng.range(0.15f, 0.55f) : 0.0f;
                ped.pos = ped.start_pt * (1.0f - ped.progress) + ped.end_pt * ped.progress;
                ped.active_crossing = (ped.progress > 0.0f);
                ped.wait_timer = rng.range(0.5f, 3.0f);
                ped.base_walk_speed = rng.range(1.65f, 2.15f);
                ped.current_speed = ped.base_walk_speed;
                ped.shirt_color = (ped_id % 2 == 0) ? Color3{0.98f, 0.78f, 0.15f} : Color3{0.15f, 0.92f, 0.85f};
                pedestrians.push_back(ped);
            }
        }

        update_360_perception_visibility();
        render_dual_cameras();
    }

    // Trigger challenging scenarios on demand ([E] button/key):
    //   0: Yellow-Light Dilemma Zone
    //   1: High-Speed Red-Light Runner Pedestrian Sprint
    //   2: Half-Parked Car / Stopped Bus Overtake Scenario (Tests Dotted-Line Safe Overtaking)
    //   3: Multi-Car Traffic Accident Ahead (Tests 360 LOS Crash Prevention & Evasive Bypass)
    void trigger_edge_case(int scenario_type) {
        int sc = scenario_type % 4;
        if (edges[ego.route_edges[ego.current_route_idx]].is_intersection_connector) {
            if (ego.current_route_idx + 1 < static_cast<int>(ego.route_edges.size())) {
                ego.current_route_idx++;
            }
            ensure_route_horizon(ego);
        }

        const LaneEdge& cur_e = edges[ego.route_edges[ego.current_route_idx]];
        int target_node = cur_e.to_node;
        Intersection& inter = intersections[target_node];
        int approach_dir = cur_e.exit_dir;
        bool is_ew = (approach_dir == 0 || approach_dir == 2);

        if (sc == 0) {
            inter.ctrl_type = IntersectionControlType::TRAFFIC_LIGHTS;
            if (!cur_e.is_intersection_connector && cur_e.samples.size() > 12) {
                size_t idx = cur_e.samples.size() - 9;
                ego.pos = cur_e.samples[idx].pos;
                ego.heading = cur_e.samples[idx].heading;
                ego.current_edge_s = cur_e.samples[idx].s;
            }
            inter.phase = is_ew ? 1 : 4;
            inter.phase_timer = 0.05f;
            ego.speed = 11.0f;
            for (auto& ped : pedestrians) {
                if (ped.intersection_id == target_node) ped.active_crossing = false;
            }
            for (auto& npc : npc_cars) {
                if ((npc.pos - ego.pos).len() < 24.0f) npc.pos += Vec2{84.0f, 84.0f};
            }
            for (auto& bike : bicycles) {
                if ((bike.pos - ego.pos).len() < 22.0f) bike.pos += Vec2{84.0f, 84.0f};
            }
        } else if (sc == 1) {
            for (auto& npc : npc_cars) {
                if ((npc.pos - ego.pos).len() < 24.0f) npc.pos += Vec2{84.0f, 84.0f};
            }
            for (auto& bike : bicycles) {
                if ((bike.pos - ego.pos).len() < 22.0f) bike.pos += Vec2{84.0f, 84.0f};
            }
            Vec2 fwd = Vec2::from_angle(ego.heading);
            Vec2 left = fwd.perp_left();
            Pedestrian jay;
            jay.id = 999;
            jay.intersection_id = target_node;
            jay.is_jaywalker = true;
            jay.is_red_light_runner = true;
            jay.active_crossing = true;
            jay.start_pt = ego.pos + fwd * 13.5f - left * 4.2f;
            jay.end_pt   = ego.pos + fwd * 13.5f + left * 4.2f;
            jay.progress = 0.30f;
            jay.pos = jay.start_pt * (1.0f - jay.progress) + jay.end_pt * jay.progress;
            jay.base_walk_speed = 3.8f;
            jay.current_speed = 3.8f;
            jay.vel = (jay.end_pt - jay.start_pt).normalized() * jay.current_speed;
            jay.shirt_color = {1.0f, 0.22f, 0.12f};
            pedestrians.push_back(jay);
        } else if (sc == 2) {
            // Scenario 2: Spawn a Half-Parked Car 16m ahead in the Dotted Overtaking Zone
            if (!cur_e.is_intersection_connector && cur_e.samples.size() > 34) {
                ego.pos = cur_e.samples[16].pos;
                ego.heading = cur_e.samples[16].heading;
                ego.current_edge_s = cur_e.samples[16].s;
                ego.speed = 8.5f;
            }
            for (auto& npc : npc_cars) {
                if ((npc.pos - ego.pos).len() < 26.0f) npc.pos += Vec2{84.0f, 84.0f};
            }
            Vec2 fwd = Vec2::from_angle(ego.heading);
            Vec2 right = fwd.perp_right();
            HalfParkedCar pk;
            pk.id = 999;
            pk.edge_id = cur_e.id;
            pk.s_pos = ego.current_edge_s + 16.0f;
            pk.pos = ego.pos + fwd * 16.0f + right * 1.15f;
            pk.heading = ego.heading;
            pk.body_color = {0.88f, 0.62f, 0.14f};
            pk.hazard_lights_on = true;
            if (!parked_cars.empty()) parked_cars[0] = pk;
            else parked_cars.push_back(pk);
        } else if (sc == 3) {
            // Scenario 3: Spawn a 2-Car Collision Wreck 15m ahead along Ego's active road
            if (!cur_e.is_intersection_connector && cur_e.samples.size() > 24) {
                ego.pos = cur_e.samples[8].pos;
                ego.heading = cur_e.samples[8].heading;
                ego.current_edge_s = cur_e.samples[8].s;
                ego.speed = 9.0f;
            }
            for (auto& npc : npc_cars) {
                if ((npc.pos - ego.pos).len() < 24.0f) npc.pos += Vec2{84.0f, 84.0f};
            }
            if (npc_cars.size() >= 2) {
                Vec2 fwd = Vec2::from_angle(ego.heading);
                Vec2 left = fwd.perp_left();
                Vec2 crash_pt = ego.pos + fwd * 14.5f;

                Vehicle& carA = npc_cars[2]; // Use non-bus NPCs
                Vehicle& carB = npc_cars[3];
                carA.pos = crash_pt - fwd * 1.1f + left * 0.45f;
                carA.heading = wrap_angle(ego.heading + 0.42f);
                carA.speed = 0.0f;
                carA.is_crashed = true;
                carA.crash_timer = 14.0f;
                carA.crash_partner_id = carB.id;

                carB.pos = crash_pt + fwd * 1.4f - left * 0.35f;
                carB.heading = wrap_angle(ego.heading - 0.58f);
                carB.speed = 0.0f;
                carB.is_crashed = true;
                carB.crash_timer = 14.0f;
                carB.crash_partner_id = carA.id;
            }
        }
        update_360_perception_visibility();
    }

    void step_pedestrians(float dt) {
        if (spontaneous_runner_cooldown > 0.0f) {
            spontaneous_runner_cooldown = std::max(0.0f, spontaneous_runner_cooldown - dt);
        }

        for (auto& ped : pedestrians) {
            const Intersection& inter = intersections[ped.intersection_id];
            bool walk_signal_green = inter.is_pedestrian_walk_green(ped.crosswalk_dir);

            if (!ped.active_crossing) {
                ped.vel = {0.0f, 0.0f};
                ped.is_red_light_runner = false;
                ped.wait_timer -= dt;
                if (ped.wait_timer > 0.0f) continue;

                if (walk_signal_green) {
                    bool clear_to_step = true;
                    if (!inter.uses_phase_signal()) {
                        Vec2 cw_mid = (ped.start_pt + ped.end_pt) * 0.5f;
                        if ((ego.pos - cw_mid).len() < 6.5f && ego.speed > 2.0f) clear_to_step = false;
                        for (const auto& npc : npc_cars) {
                            if ((npc.pos - cw_mid).len() < 6.5f && npc.speed > 2.0f) clear_to_step = false;
                        }
                    }
                    if (clear_to_step) {
                        ped.active_crossing = true;
                        ped.is_red_light_runner = false;
                        ped.current_speed = ped.base_walk_speed;
                    }
                } else {
                    Vec2 cw_mid = (ped.start_pt + ped.end_pt) * 0.5f;
                    float dist_ego = (cw_mid - ego.pos).len();
                    Vec2 ego_fwd = Vec2::from_angle(ego.heading);
                    bool ego_approaching = ((cw_mid - ego.pos).dot(ego_fwd) > 9.0f && dist_ego > 11.5f && dist_ego < 20.5f && ego.speed > 4.5f);

                    if (ego_approaching && spontaneous_runner_cooldown <= 0.0f && rng.range_int(0, 1000) < 18) {
                        ped.active_crossing = true;
                        ped.is_red_light_runner = true;
                        ped.current_speed = rng.range(3.4f, 4.4f);
                        spontaneous_runner_cooldown = 14.0f;
                    }
                }
                continue;
            }

            if (ped.is_red_light_runner) {
                ped.current_speed = std::max(ped.current_speed, 3.6f);
            } else if (!walk_signal_green) {
                ped.current_speed = ped.base_walk_speed * 1.45f;
            } else {
                ped.current_speed = ped.base_walk_speed;
            }

            ped.anim_phase += dt * ped.current_speed * 4.2f;
            float dist = std::max(0.1f, (ped.end_pt - ped.start_pt).len());
            ped.progress += (ped.current_speed * dt) / dist;

            if (ped.progress >= 1.0f) {
                std::swap(ped.start_pt, ped.end_pt);
                ped.progress = 0.0f;
                ped.active_crossing = false;
                ped.is_red_light_runner = false;
                ped.wait_timer = rng.range(4.0f, 8.5f);
                ped.pos = ped.start_pt;
                ped.vel = {0.0f, 0.0f};
            } else {
                ped.pos = ped.start_pt * (1.0f - ped.progress) + ped.end_pt * ped.progress;
                ped.vel = (ped.end_pt - ped.start_pt).normalized() * ped.current_speed;
            }
        }
    }

    void step_bicycles(float dt) {
        for (auto& bike : bicycles) {
            ensure_bicycle_route(bike);

            float best_d2 = 1e9f;
            int best_slot = bike.current_route_idx;
            float best_s = bike.current_edge_s;

            int search_end = std::min(static_cast<int>(bike.route_edges.size()), bike.current_route_idx + 2);
            for (int slot = bike.current_route_idx; slot < search_end; ++slot) {
                const LaneEdge& e = edges[bike.route_edges[slot]];
                for (const auto& ps : e.samples) {
                    Vec2 right_d = Vec2::from_angle(ps.heading).perp_right();
                    Vec2 lane_pt = ps.pos + right_d * bike.lateral_offset;
                    float d2 = (lane_pt - bike.pos).len_sq();
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_slot = slot;
                        best_s = ps.s;
                    }
                }
            }
            if (best_slot > bike.current_route_idx) {
                bike.current_route_idx = best_slot;
                ensure_bicycle_route(bike);
            }
            bike.current_edge_s = best_s;

            const LaneEdge& active_e = edges[bike.route_edges[bike.current_route_idx]];
            float target_s = bike.current_edge_s + 2.6f;
            Vec2 target_pt = bike.pos + Vec2::from_angle(bike.heading) * 2.6f;

            if (target_s <= active_e.length) {
                PathSample ps = active_e.eval_at_s(target_s);
                target_pt = ps.pos + Vec2::from_angle(ps.heading).perp_right() * bike.lateral_offset;
            } else if (bike.current_route_idx + 1 < static_cast<int>(bike.route_edges.size())) {
                const LaneEdge& next_e = edges[bike.route_edges[bike.current_route_idx + 1]];
                float rem_s = target_s - active_e.length;
                PathSample ps = next_e.eval_at_s(rem_s);
                target_pt = ps.pos + Vec2::from_angle(ps.heading).perp_right() * bike.lateral_offset;
            }

            Vec2 b_fwd = Vec2::from_angle(bike.heading);
            Vec2 b_left = b_fwd.perp_left();
            Vec2 to_target = target_pt - bike.pos;
            float local_y = to_target.dot(b_left);
            float dist_sq = std::max(0.8f, to_target.len_sq());
            float desired_steer = clampf(std::atan(2.0f * 1.08f * local_y / dist_sq), -0.65f, 0.65f);
            bike.steer_angle = lerpf(bike.steer_angle, desired_steer, clampf(dt * 10.0f, 0.0f, 1.0f));

            float v_target = bike.cruise_speed;
            if (!active_e.is_intersection_connector) {
                float d_stop = active_e.length - bike.current_edge_s;
                const Intersection& inter = intersections[active_e.to_node];
                if (inter.uses_phase_signal()) {
                    LightState ls = inter.get_light_for_approach(active_e.exit_dir);
                    if ((ls == LightState::RED || ls == LightState::YELLOW) && d_stop < 6.0f) {
                        v_target = (d_stop < 2.4f) ? 0.0f : 1.6f;
                    }
                }
            }

            for (const auto& ped : pedestrians) {
                if (!ped.active_crossing) continue;
                Vec2 rel = ped.pos - bike.pos;
                if (rel.dot(b_fwd) > 0.0f && rel.len() < 5.5f) {
                    v_target = 0.0f;
                }
            }

            bike.speed = lerpf(bike.speed, v_target, clampf(dt * 4.5f, 0.0f, 1.0f));
            bike.pedal_phase += dt * bike.speed * 3.4f;

            float yaw_rate = (bike.speed / 1.08f) * std::tan(bike.steer_angle);
            bike.pos += b_fwd * (bike.speed * dt);
            bike.heading = wrap_angle(bike.heading + yaw_rate * dt);
        }
    }

    // NPC Traffic Driver (Includes City Transit Buses stopping at "BUS" zones & NPC Overtaking of Half-Parked Cars)
    void step_npc_vehicles(float dt) {
        if (spontaneous_accident_cooldown > 0.0f) {
            spontaneous_accident_cooldown = std::max(0.0f, spontaneous_accident_cooldown - dt);
        }

        for (auto& npc : npc_cars) {
            if (npc.is_crashed) {
                npc.crash_timer -= dt;
                npc.step_physics(dt);
                if (npc.crash_timer <= 0.0f) {
                    npc.is_crashed = false;
                    npc.is_reckless_driver = false;
                    npc.crash_partner_id = -1;
                    const LaneEdge& ce = edges[npc.route_edges[npc.current_route_idx]];
                    for (const auto& ps : ce.samples) {
                        if (ps.s >= npc.current_edge_s) {
                            npc.pos = ps.pos;
                            npc.heading = ps.heading;
                            break;
                        }
                    }
                    npc.speed = 4.5f;
                }
            }
        }

        if (spontaneous_accident_cooldown <= 0.0f && active_accident_count == 0 && rng.range_int(0, 1000) < 7) {
            for (size_t i = 0; i < npc_cars.size(); ++i) {
                if (npc_cars[i].is_crashed || npc_cars[i].is_bus) continue;
                for (size_t j = i + 1; j < npc_cars.size(); ++j) {
                    if (npc_cars[j].is_crashed || npc_cars[j].is_bus) continue;
                    float d_pair = (npc_cars[i].pos - npc_cars[j].pos).len();
                    if (d_pair > 4.5f && d_pair < 11.5f) {
                        npc_cars[i].is_reckless_driver = true;
                        spontaneous_accident_cooldown = 28.0f;
                        break;
                    }
                }
                if (spontaneous_accident_cooldown > 0.0f) break;
            }
        }

        for (size_t idx = 0; idx < npc_cars.size(); ++idx) {
            Vehicle& npc = npc_cars[idx];
            if (npc.is_crashed) continue;

            update_vehicle_route_progress(npc);

            std::vector<PathSample> lookahead;
            get_route_lookahead(npc, 22.0f, lookahead);
            if (lookahead.empty()) continue;

            const LaneEdge& cur_e = edges[npc.route_edges[npc.current_route_idx]];
            bool inside_inter = cur_e.is_intersection_connector;
            Vec2 fwd = Vec2::from_angle(npc.heading);
            Vec2 left = fwd.perp_left();

            float v_target = npc.is_bus ? 7.8f : (npc.is_reckless_driver ? 12.5f : 9.4f);
            float target_lat_shift = 0.0f;

            // A. City Transit Bus: Stop at Designated "BUS" Pavement Zones for 6.0 seconds
            npc.is_stopped_at_bus_zone = false;
            if (npc.is_bus && !inside_inter) {
                for (const auto& bs : bus_stops) {
                    if (bs.edge_id == cur_e.id && npc.last_serviced_bus_stop_id != bs.id) {
                        float dist_to_bs = bs.s_center - npc.current_edge_s;
                        if (dist_to_bs > -1.5f && dist_to_bs < 16.0f) {
                            target_lat_shift = -0.55f; // Pull slightly right into the BUS bay box
                            if (dist_to_bs < 2.2f) {
                                v_target = 0.0f;
                                npc.is_stopped_at_bus_zone = true;
                                npc.bus_stop_dwell_timer += dt;
                                if (npc.bus_stop_dwell_timer >= 6.0f) {
                                    npc.last_serviced_bus_stop_id = bs.id;
                                    npc.bus_stop_dwell_timer = 0.0f;
                                    npc.is_stopped_at_bus_zone = false;
                                }
                            } else {
                                v_target = std::min(v_target, std::sqrt(2.0f * 2.2f * dist_to_bs));
                            }
                            break;
                        }
                    }
                }
            }

            // B. Check for Half-Parked Cars or Stopped Buses ahead -> Overtake on the Left ONLY when safe & legal
            bool npc_waiting_to_overtake = false;
            if (!inside_inter && !npc.is_bus) {
                bool npc_needs_overtake = false;
                float npc_obs_ds = 99.0f;
                PathSample npc_road_ps = cur_e.eval_at_s(npc.current_edge_s);
                Vec2 road_fwd  = Vec2::from_angle(npc_road_ps.heading);
                Vec2 road_left = road_fwd.perp_left();

                for (const auto& pk : parked_cars) {
                    if (pk.edge_id == cur_e.id) {
                        float ds = pk.s_pos - npc.current_edge_s;
                        if (ds > -6.5f && ds < 20.0f && ds < npc_obs_ds) {
                            npc_needs_overtake = true;
                            npc_obs_ds = ds;
                        }
                    }
                }
                // If Ego is parked in a Lane-1 lateral parking space on this multi-lane road, overtake Ego cleanly via Lane 2
                if (ego.gear == GearState::PARK && !ego.route_edges.empty() &&
                    ego.route_edges[ego.current_route_idx] == cur_e.id) {
                    Vec2 rel_ego = ego.pos - npc_road_ps.pos;
                    float ds_ego = rel_ego.dot(road_fwd);
                    float lat_ego = std::fabs(rel_ego.dot(road_left));
                    if (lat_ego < 1.35f && ds_ego > -6.5f && ds_ego < 20.0f && ds_ego < npc_obs_ds) {
                        npc_needs_overtake = true;
                        npc_obs_ds = ds_ego;
                    }
                }
                for (size_t j = 0; j < npc_cars.size(); ++j) {
                    if (j == idx) continue;
                    const Vehicle& other = npc_cars[j];
                    if ((other.is_stopped_at_bus_zone || other.is_crashed) &&
                        other.route_edges[other.current_route_idx] == cur_e.id) {
                        float ds = other.current_edge_s - npc.current_edge_s;
                        float clear_rear = other.is_bus ? -9.2f : -6.5f;
                        if (ds > clear_rear && ds < 21.0f && ds < npc_obs_ds) {
                            npc_needs_overtake = true;
                            npc_obs_ds = ds;
                        }
                    }
                }

                if (npc_needs_overtake) {
                    bool already_alongside = (npc.current_lat_offset > 1.25f && npc_obs_ds < 4.8f);
                    float rem_road = cur_e.length - npc.current_edge_s;
                    bool legal_zone = (cur_e.is_dotted_overtake_zone(npc.current_edge_s) && rem_road > 14.0f) || already_alongside;
                    float req_oncoming_d = clampf((std::max(0.0f, npc_obs_ds) + 12.0f) * 2.35f, 30.0f, 56.0f);
                    bool pass_lane_safe = true;
                    float max_oncoming_lat = (cur_e.num_lanes_same_dir >= 2) ? 3.85f : 6.6f;

                    auto check_other_conflict = [&](const Vehicle& other) {
                        if (other.is_crashed || other.is_stopped_at_bus_zone) return;
                        if (other.is_ego && other.gear == GearState::PARK) return;
                        Vec2 rel_road = other.pos - npc_road_ps.pos;
                        float road_lon = rel_road.dot(road_fwd);
                        float road_lat = rel_road.dot(road_left);
                        Vec2 ofwd = Vec2::from_angle(other.heading);
                        float h_dot = ofwd.dot(road_fwd);

                        // 1. Oncoming traffic check in left passing/opposing lane
                        if (h_dot < -0.30f) {
                            if (road_lon > -4.0f && road_lon < req_oncoming_d && road_lat > 0.4f && road_lat < max_oncoming_lat) {
                                bool npc_tie_break = (!other.is_ego && other.speed < 0.45f && !other.is_overtaking &&
                                                      road_lat > 1.45f && road_lon > std::max(9.5f, npc_obs_ds + 5.5f) &&
                                                      npc.yield_impatience > 1.8f && npc.id < other.id);
                                if (!npc_tie_break) {
                                    pass_lane_safe = false;
                                }
                            }
                        }
                        // 2. Left-lane blind-spot / passing lane occupancy check (same direction)
                        if (h_dot > 0.30f) {
                            if (road_lon > -15.0f && road_lon < 22.0f && road_lat > 0.85f && road_lat < 4.8f) {
                                pass_lane_safe = false;
                            }
                        }
                    };

                    if (!already_alongside) {
                        check_other_conflict(ego);
                        for (size_t j = 0; j < npc_cars.size(); ++j) {
                            if (j != idx) check_other_conflict(npc_cars[j]);
                        }
                    }

                    if (legal_zone && pass_lane_safe) {
                        target_lat_shift = +2.55f;
                        // If starting an overtake from a standstill close behind the obstacle, steer into Lane 2 before accelerating past it
                        if (npc_obs_ds > 0.0f && npc_obs_ds < npc.length + 2.2f && npc.current_lat_offset < 1.35f) {
                            v_target = std::min(v_target, 2.2f);
                        }
                    } else {
                        // Not safe to overtake. Stay in Lane 1 and brake smoothly behind the obstacle
                        target_lat_shift = 0.0f;
                        npc_waiting_to_overtake = true;
                        if (npc_obs_ds > 0.0f) {
                            float safe_gap = std::max(0.0f, npc_obs_ds - npc.length - 2.4f);
                            float v_wait = (safe_gap < 2.0f) ? 0.0f : std::sqrt(2.0f * 3.2f * safe_gap);
                            v_target = std::min(v_target, v_wait);
                        }
                    }
                }
            }
            npc.current_lat_offset = lerpf(npc.current_lat_offset, target_lat_shift, clampf(dt * 3.8f, 0.0f, 1.0f));
            npc.is_overtaking = (npc.current_lat_offset > 0.6f);

            // Pure pursuit steering with lateral overtaking offset (tighter chord inside roundabouts/intersections)
            float ld = inside_inter ? clampf(2.2f + 0.18f * npc.speed, 2.2f, 3.4f)
                                    : clampf(3.4f + 0.42f * npc.speed, 3.4f, 8.5f);
            Vec2 target_pt = lookahead.back().pos;
            float target_h = lookahead.back().heading;
            for (const auto& ps : lookahead) {
                if (ps.s >= ld) {
                    target_pt = ps.pos;
                    target_h = ps.heading;
                    break;
                }
            }
            target_pt += Vec2::from_angle(target_h).perp_left() * npc.current_lat_offset;

            Vec2 to_target = target_pt - npc.pos;
            float local_y = to_target.dot(left);
            float dist_sq = std::max(1.0f, to_target.len_sq());
            float steer_rad = std::atan(2.0f * npc.wheelbase * local_y / dist_sq);
            npc.steer_cmd = clampf(steer_rad / 0.58f, -1.0f, 1.0f);

            for (const auto& ps : lookahead) {
                if (!npc.is_reckless_driver && ps.s < 13.0f && std::fabs(ps.curvature) > 0.05f) {
                    v_target = std::min(v_target, 5.2f);
                }
            }

            if (!inside_inter && !npc.is_reckless_driver) {
                float d_stop = cur_e.length - npc.current_edge_s;
                const Intersection& inter = intersections[cur_e.to_node];
                int app_dir = cur_e.exit_dir;

                if (inter.uses_phase_signal()) {
                    LightState ls = inter.get_light_for_approach(app_dir);
                    if ((ls == LightState::RED || ls == LightState::YELLOW) && d_stop < 18.0f) {
                        if (d_stop <= 3.0f) v_target = 0.0f;
                        else v_target = std::min(v_target, std::sqrt(2.0f * 2.8f * (d_stop - 2.6f)));
                    }
                } else if (inter.has_stop_sign_for_approach(app_dir)) {
                    if (!npc.stop_sign_completed_for_edge && d_stop < 15.0f) {
                        if (d_stop <= 3.4f) {
                            v_target = 0.0f;
                            if (npc.speed < 0.35f) {
                                npc.stop_sign_hold_timer += dt;
                                if (npc.stop_sign_hold_timer >= 0.65f) {
                                    npc.stop_sign_completed_for_edge = true;
                                }
                            }
                        } else {
                            v_target = std::min(v_target, std::sqrt(2.0f * 2.6f * (d_stop - 2.8f)));
                        }
                    }
                } else if (inter.has_yield_sign_for_approach(app_dir)) {
                    if (d_stop < 14.0f) {
                        bool major_busy = false;
                        auto check_major = [&](const Vehicle& other) {
                            float d_inter = (other.pos - inter.center).len();
                            if (d_inter < 18.0f && other.speed > 1.0f) {
                                Vec2 ofwd = Vec2::from_angle(other.heading);
                                if (std::fabs(ofwd.dot(fwd)) < 0.45f && (inter.center - other.pos).dot(ofwd) > -2.0f) {
                                    major_busy = true;
                                }
                            }
                        };
                        check_major(ego);
                        for (size_t j = 0; j < npc_cars.size() && !major_busy; ++j) {
                            if (j != idx) check_major(npc_cars[j]);
                        }
                        if (major_busy && npc.yield_impatience < 3.5f) {
                            if (d_stop <= 3.2f) v_target = 0.0f;
                            else v_target = std::min(v_target, std::sqrt(2.0f * 2.6f * (d_stop - 2.8f)));
                        }
                    }
                }

                if (v_target > 0.1f && d_stop < 7.5f && npc.active_intent == TurnIntent::LEFT) {
                    int opp_dir = (app_dir + 2) & 3;
                    Vec2 inter_c = inter.center;
                    Vec2 opp_v = dir_vec(opp_dir);

                    auto check_oncoming = [&](const Vehicle& other) {
                        if ((other.pos - inter_c).len() > 20.0f) return false;
                        Vec2 ofwd = Vec2::from_angle(other.heading);
                        if (ofwd.dot(opp_v) > 0.65f && other.active_intent != TurnIntent::LEFT) {
                            if ((inter_c - other.pos).dot(ofwd) > -3.0f) return true;
                        }
                        return false;
                    };

                    bool oncoming_busy = check_oncoming(ego);
                    for (size_t j = 0; j < npc_cars.size() && !oncoming_busy; ++j) {
                        if (j != idx) oncoming_busy = check_oncoming(npc_cars[j]);
                    }
                    if (oncoming_busy && npc.yield_impatience < 3.5f) {
                        if (d_stop <= 3.2f) v_target = 0.0f;
                        else v_target = std::min(v_target, 2.5f);
                    }
                }
            }

            if (!npc.is_reckless_driver) {
                auto evaluate_vehicle_interaction = [&](const Vehicle& other) {
                    if (npc.is_overtaking && (other.is_stopped_at_bus_zone || other.is_crashed || (other.is_ego && other.gear == GearState::PARK))) return;

                    float direct_dist = (other.pos - npc.pos).len();
                    if (direct_dist > 22.0f) return;

                    Vec2 rel = other.pos - npc.pos;
                    float lon = rel.dot(fwd);
                    if (lon < -0.5f) return;

                    Vec2 other_fwd = Vec2::from_angle(other.heading);
                    float heading_align = fwd.dot(other_fwd);

                    // If Ego is actively executing a multi-point parking/unparking or reverse maneuver ahead on this corridor
                    // (NOT already parked in GearState::PARK), politely yield so no NPC drives into Ego while backing in
                    if (other.is_ego && other.gear != GearState::PARK && (other.gear == GearState::REVERSE || other.speed < 2.2f)) {
                        if (lon > 1.0f && lon < 14.5f && std::fabs(rel.dot(left)) < 2.35f) {
                            if (lon < 8.5f) v_target = 0.0f;
                            else v_target = std::min(v_target, 1.8f);
                            return;
                        }
                    }
                    if (heading_align < -0.55f && !inside_inter) {
                        if (lon > 0.5f && lon < 22.0f && std::fabs(rel.dot(left)) < 1.95f) {
                            if (lon < 10.5f) v_target = 0.0f;
                            else v_target = std::min(v_target, 2.2f);
                        }
                        return;
                    }

                    float conflict_s = 99.0f;
                    for (const auto& ps : lookahead) {
                        if (ps.s < 0.8f || ps.s > 13.0f) continue;
                        Vec2 ps_shifted = ps.pos + Vec2::from_angle(ps.heading).perp_left() * npc.current_lat_offset;
                        if ((other.pos - ps_shifted).len() < 1.95f) {
                            conflict_s = ps.s;
                            break;
                        }
                    }
                    if (conflict_s > 13.0f) return;

                    if (heading_align > 0.45f) {
                        if (lon > 0.5f && std::fabs(rel.dot(left)) < 1.75f) {
                            float min_clear = (npc.is_bus || other.is_bus) ? 8.2f : 5.8f;
                            if (conflict_s < min_clear) v_target = 0.0f;
                            else v_target = std::min(v_target, other.speed * ((conflict_s - (min_clear - 0.6f)) / 7.0f));
                        }
                    } else {
                        const LaneEdge& other_e = edges[other.route_edges[other.current_route_idx]];
                        bool other_in_inter = other_e.is_intersection_connector;

                        bool i_have_priority = false;
                        if (other.is_ego) {
                            i_have_priority = (inside_inter && !other_in_inter && npc.current_edge_s > 3.0f);
                        } else if (other.is_crashed) {
                            i_have_priority = false;
                        } else if (inside_inter != other_in_inter) {
                            i_have_priority = inside_inter;
                        } else if (inside_inter && other_in_inter) {
                            if (std::fabs(npc.current_edge_s - other.current_edge_s) > 0.9f) {
                                i_have_priority = (npc.current_edge_s > other.current_edge_s);
                            } else if ((npc.active_intent == TurnIntent::LEFT) != (other.active_intent == TurnIntent::LEFT)) {
                                i_have_priority = (npc.active_intent != TurnIntent::LEFT);
                            } else {
                                i_have_priority = (npc.id < other.id);
                            }
                        } else {
                            i_have_priority = (npc.id < other.id);
                        }

                        if (!i_have_priority) {
                            if (conflict_s < 6.2f) v_target = 0.0f;
                            else v_target = std::min(v_target, 2.5f);
                        } else if (lon > 1.2f && direct_dist < 3.5f && std::fabs(rel.dot(left)) < 1.35f) {
                            v_target = 0.0f;
                        }
                    }
                };

                evaluate_vehicle_interaction(ego);
                for (size_t j = 0; j < npc_cars.size(); ++j) {
                    if (j != idx) evaluate_vehicle_interaction(npc_cars[j]);
                }

                for (const auto& bike : bicycles) {
                    Vec2 rel = bike.pos - npc.pos;
                    float lon = rel.dot(fwd);
                    float lat = std::fabs(rel.dot(left));
                    if (lon > 0.5f && lon < 10.0f && lat < 1.55f) {
                        v_target = std::min(v_target, bike.speed);
                    }
                }
                for (const auto& ped : pedestrians) {
                    if (!ped.active_crossing) continue;
                    Vec2 rel = ped.pos - npc.pos;
                    float lon = rel.dot(fwd);
                    float lat = std::fabs(rel.dot(left));
                    if (lon > 0.0f && lon < 9.5f && lat < 2.4f) {
                        v_target = 0.0f;
                    }
                }
            }

            // Anti-Gridlock Watchdog (Paused while bus is legally dwelling at a BUS stop)
            LightState cur_light = inside_inter
                ? LightState::GREEN
                : intersections[cur_e.to_node].get_light_for_approach(cur_e.exit_dir);

            if (!npc.is_stopped_at_bus_zone && npc.speed < 0.35f && (inside_inter || cur_light == LightState::GREEN)) {
                npc.yield_impatience += dt;
                // Never force an NPC waiting behind a physical lane obstacle in Lane 1 to drive straight into the obstacle
                if (npc.yield_impatience > 2.2f && !npc_waiting_to_overtake) {
                    v_target = std::max(v_target, 4.5f);
                }
            } else if (npc.speed > 1.5f) {
                npc.yield_impatience = 0.0f;
            }

            float v_err = v_target - npc.speed;
            npc.accel_cmd = clampf(v_err * 0.48f, -1.0f, 0.80f);
            npc.step_physics(dt);

            if (npc.is_reckless_driver) {
                for (size_t j = 0; j < npc_cars.size(); ++j) {
                    if (j == idx || npc_cars[j].is_crashed || npc_cars[j].is_bus) continue;
                    if ((npc.pos - npc_cars[j].pos).len() < 3.8f) {
                        npc.is_crashed = true;
                        npc.is_reckless_driver = false;
                        npc.crash_timer = 14.0f;
                        npc.crash_partner_id = npc_cars[j].id;
                        npc.heading = wrap_angle(npc.heading + 0.38f);
                        npc.speed = 0.0f;

                        npc_cars[j].is_crashed = true;
                        npc_cars[j].crash_timer = 14.0f;
                        npc_cars[j].crash_partner_id = npc.id;
                        npc_cars[j].heading = wrap_angle(npc_cars[j].heading - 0.44f);
                        npc_cars[j].speed = 0.0f;
                        break;
                    }
                }
            }
        }

        update_360_perception_visibility();
    }

    void render_dual_cameras() {
        render_single_camera(0, 52.0f * DEG2RAD, 0.06f);
        render_single_camera(1, 106.0f * DEG2RAD, 0.14f);
    }

private:
    void render_single_camera(int cam_idx, float fov_x_rad, float pitch_down_rad) {
        float* ch_r = &camera_tensor[(cam_idx * 3 + 0) * CAM_W * CAM_H];
        float* ch_g = &camera_tensor[(cam_idx * 3 + 1) * CAM_W * CAM_H];
        float* ch_b = &camera_tensor[(cam_idx * 3 + 2) * CAM_W * CAM_H];
        std::array<float, CAM_W * CAM_H> zbuf;
        zbuf.fill(1e6f);

        Vec2 fwd2 = Vec2::from_angle(ego.heading);
        Vec2 left2 = fwd2.perp_left();
        Vec3 cam_pos = {ego.pos.x + fwd2.x * 0.6f, ego.pos.y + fwd2.y * 0.6f, 1.45f};

        float cos_p = std::cos(pitch_down_rad);
        float sin_p = std::sin(pitch_down_rad);
        Vec3 cam_fwd   = {fwd2.x * cos_p, fwd2.y * cos_p, -sin_p};
        Vec3 cam_right = {-left2.x, -left2.y, 0.0f};
        Vec3 cam_up    = {fwd2.x * sin_p, fwd2.y * sin_p, cos_p};

        float focal_x = (CAM_W * 0.5f) / std::tan(fov_x_rad * 0.5f);
        float focal_y = focal_x;

        for (int py = 0; py < CAM_H; ++py) {
            for (int px = 0; px < CAM_W; ++px) {
                float ndc_x = (px + 0.5f - CAM_W * 0.5f) / focal_x;
                float ndc_y = (CAM_H * 0.5f - (py + 0.5f)) / focal_y;
                Vec3 ray = (cam_fwd + cam_right * ndc_x + cam_up * ndc_y).normalized();

                int idx = py * CAM_W + px;
                if (ray.z >= -0.01f) {
                    float t = clampf(ray.z * 2.5f, 0.0f, 1.0f);
                    ch_r[idx] = lerpf(0.14f, 0.06f, t);
                    ch_g[idx] = lerpf(0.20f, 0.10f, t);
                    ch_b[idx] = lerpf(0.32f, 0.22f, t);
                } else {
                    float t_ground = -cam_pos.z / ray.z;
                    zbuf[idx] = t_ground;
                    Vec2 wp = {cam_pos.x + ray.x * t_ground, cam_pos.y + ray.y * t_ground};

                    Color3 ground_col = sample_ground_surface(wp);
                    float fog = clampf(t_ground / 75.0f, 0.0f, 0.65f);
                    ch_r[idx] = lerpf(ground_col.r, 0.12f, fog);
                    ch_g[idx] = lerpf(ground_col.g, 0.16f, fog);
                    ch_b[idx] = lerpf(ground_col.b, 0.24f, fog);
                }
            }
        }

        auto splat_3d_box = [&](Vec3 center, float half_w, float half_h, float depth_bias, Color3 col) {
            Vec3 rel = center - cam_pos;
            float cz = rel.dot(cam_fwd);
            if (cz < 0.6f || cz > 65.0f) return;
            float cx = rel.dot(cam_right);
            float cy = rel.dot(cam_up);

            float sx = (CAM_W * 0.5f) + (cx / cz) * focal_x;
            float sy = (CAM_H * 0.5f) - (cy / cz) * focal_y;
            float rad_x = std::max(0.75f, (half_w / cz) * focal_x);
            float rad_y = std::max(0.75f, (half_h / cz) * focal_y);

            int x0 = std::max(0, static_cast<int>(std::floor(sx - rad_x)));
            int x1 = std::min(CAM_W - 1, static_cast<int>(std::ceil(sx + rad_x)));
            int y0 = std::max(0, static_cast<int>(std::floor(sy - rad_y)));
            int y1 = std::min(CAM_H - 1, static_cast<int>(std::ceil(sy + rad_y)));

            float eff_z = cz + depth_bias;
            for (int py = y0; py <= y1; ++py) {
                for (int px = x0; px <= x1; ++px) {
                    int idx = py * CAM_W + px;
                    if (eff_z < zbuf[idx]) {
                        zbuf[idx] = eff_z;
                        ch_r[idx] = col.r;
                        ch_g[idx] = col.g;
                        ch_b[idx] = col.b;
                    }
                }
            }
        };

        std::vector<PathSample> route_pts;
        get_route_lookahead(ego, 28.0f, route_pts);
        for (size_t i = 0; i < route_pts.size(); i += 2) {
            Vec2 p = route_pts[i].pos;
            splat_3d_box({p.x, p.y, 0.08f}, 0.35f, 0.08f, -0.3f, {0.10f, 0.85f, 0.95f});
        }

        for (const auto& inter : intersections) {
            float d2 = (inter.center - ego.pos).len_sq();
            if (d2 > 55.0f * 55.0f) continue;

            if (inter.is_police_controlled()) {
                splat_3d_box({inter.center.x, inter.center.y, 1.0f}, 0.55f, 0.95f, -0.1f, {0.15f, 0.35f, 0.92f});
            }

            for (int arm = 0; arm < 4; ++arm) {
                if (!inter.is_arm_active(arm)) continue;
                Vec2 arm_d = dir_vec(arm);
                Vec2 right_d = arm_d.perp_right();

                if (inter.ctrl_type == IntersectionControlType::TRAFFIC_LIGHTS) {
                    Vec2 pole_xy = inter.center + arm_d * (inter.half_size + 0.5f) + right_d * (LANE_OFFSET + 0.5f);
                    LightState ls = inter.get_light_for_approach(arm);
                    Color3 light_col = (ls == LightState::RED)    ? Color3{1.0f, 0.08f, 0.08f} :
                                       (ls == LightState::YELLOW) ? Color3{1.0f, 0.88f, 0.05f} :
                                                                    Color3{0.08f, 0.98f, 0.22f};
                    splat_3d_box({pole_xy.x, pole_xy.y, 3.3f}, 0.55f, 0.85f, 0.0f, {0.12f, 0.12f, 0.15f});
                    splat_3d_box({pole_xy.x, pole_xy.y, 3.3f}, 0.42f, 0.45f, -0.15f, light_col);
                } else {
                    Vec2 sign_xy = inter.center - arm_d * (inter.half_size + 0.6f) + right_d * (LANE_WIDTH + 0.65f);
                    if (inter.has_stop_sign_for_approach(arm)) {
                        splat_3d_box({sign_xy.x, sign_xy.y, 2.1f}, 0.42f, 0.42f, -0.1f, {0.95f, 0.12f, 0.12f});
                    } else if (inter.has_yield_sign_for_approach(arm)) {
                        splat_3d_box({sign_xy.x, sign_xy.y, 2.1f}, 0.42f, 0.42f, -0.1f, {0.98f, 0.72f, 0.10f});
                    }
                }
            }
        }

        // Half-parked cars in camera tensor
        for (const auto& pk : parked_cars) {
            float d2 = (pk.pos - ego.pos).len_sq();
            if (d2 > 55.0f * 55.0f) continue;
            splat_3d_box({pk.pos.x, pk.pos.y, 0.82f}, 1.10f, 0.75f, 0.0f, pk.body_color);
        }

        for (const auto& npc : npc_cars) {
            float d2 = (npc.pos - ego.pos).len_sq();
            if (d2 > 55.0f * 55.0f) continue;

            Color3 bcol = npc.is_crashed ? Color3{1.0f, 0.35f, 0.10f} : npc.body_color;
            float hw = npc.is_bus ? 1.45f : 1.15f;
            float hh = npc.is_bus ? 1.45f : 0.78f;
            splat_3d_box({npc.pos.x, npc.pos.y, hh}, hw, hh, 0.0f, bcol);

            Vec2 npc_fwd = Vec2::from_angle(npc.heading);
            Vec2 rear = npc.pos - npc_fwd * (npc.length * 0.46f);
            Color3 tail_col = (npc.braking_lights || npc.is_crashed || npc.is_stopped_at_bus_zone)
                            ? Color3{1.0f, 0.05f, 0.05f} : Color3{0.55f, 0.08f, 0.08f};
            splat_3d_box({rear.x, rear.y, 0.82f}, 0.85f, 0.28f, -0.2f, tail_col);
        }

        for (const auto& bike : bicycles) {
            float d2 = (bike.pos - ego.pos).len_sq();
            if (d2 > 45.0f * 45.0f) continue;
            splat_3d_box({bike.pos.x, bike.pos.y, 0.95f}, 0.45f, 0.75f, -0.1f, bike.rider_color);
        }

        for (const auto& ped : pedestrians) {
            float d2 = (ped.pos - ego.pos).len_sq();
            if (d2 > 45.0f * 45.0f) continue;
            Color3 pcol = ped.is_red_light_runner ? Color3{1.0f, 0.15f, 0.15f} : ped.shirt_color;
            splat_3d_box({ped.pos.x, ped.pos.y, 0.90f}, 0.42f, 0.85f, -0.1f, pcol);
            splat_3d_box({ped.pos.x, ped.pos.y, 1.65f}, 0.25f, 0.25f, -0.12f, {0.95f, 0.80f, 0.68f});
        }
    }

    Color3 sample_ground_surface(Vec2 wp) const {
        float best_dx = 1e6f;
        float best_dy = 1e6f;
        for (int i = 0; i < GRID_N; ++i) {
            float rx = (i - 1) * BLOCK_SPACING;
            float ry = (i - 1) * BLOCK_SPACING;
            if (std::fabs(wp.x - rx) < best_dx) best_dx = std::fabs(wp.x - rx);
            if (std::fabs(wp.y - ry) < best_dy) best_dy = std::fabs(wp.y - ry);
        }

        constexpr float ROAD_HALF = LANE_WIDTH + 0.35f;
        bool on_v_road = (best_dx <= ROAD_HALF);
        bool on_h_road = (best_dy <= ROAD_HALF);

        if (!on_v_road && !on_h_road) {
            if (best_dx <= ROAD_HALF + 1.4f || best_dy <= ROAD_HALF + 1.4f) {
                return {0.42f, 0.43f, 0.46f};
            }
            return {0.14f, 0.24f, 0.16f};
        }

        if (best_dx <= INTER_HALF && best_dy <= INTER_HALF) {
            bool in_v_cw = (best_dy >= INTER_HALF - 2.2f && best_dy <= INTER_HALF - 0.3f && on_v_road);
            bool in_h_cw = (best_dx >= INTER_HALF - 2.2f && best_dx <= INTER_HALF - 0.3f && on_h_road);
            if (in_v_cw) {
                int stripe = static_cast<int>(std::floor((wp.x + 200.0f) * 1.6f));
                if (stripe % 2 == 0) return {0.90f, 0.92f, 0.95f};
            }
            if (in_h_cw) {
                int stripe = static_cast<int>(std::floor((wp.y + 200.0f) * 1.6f));
                if (stripe % 2 == 0) return {0.90f, 0.92f, 0.95f};
            }
            return {0.20f, 0.21f, 0.24f};
        }

        float dist_from_center = on_v_road ? best_dx : best_dy;
        if (dist_from_center < 0.22f) {
            return {0.95f, 0.78f, 0.12f};
        }
        if (std::fabs(dist_from_center - (ROAD_HALF - 0.22f)) < 0.20f) {
            return {0.92f, 0.92f, 0.95f};
        }
        return {0.20f, 0.21f, 0.24f};
    }
};

} // namespace sim
