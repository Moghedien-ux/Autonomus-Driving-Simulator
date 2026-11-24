#pragma once
// Rule-based driving planner: pure-pursuit + Stanley path tracking, intersection logic,
// multi-lane overtaking, collision avoidance, and multi-layout autopark.
#include "city_world.hpp"

namespace sim {

enum class AutoparkState : int {
    INACTIVE = 0,
    SEARCHING_360_LOS = 1,
    APPROACHING_AISLE = 2,
    SETUP_POSITION = 3,
    MANEUVER_INTO_STALL = 4,
    PARKED_IN_BOUNDS = 5,
    UNPARKING_EXIT = 6
};

struct RuleTelemetry {
    bool rule_pure_pursuit = true;
    bool rule_corner_grip_limit = false;
    bool rule_red_light_stop = false;
    bool rule_dilemma_stop = false;
    bool rule_dilemma_commit = false;
    bool rule_stop_sign = false;
    bool rule_yield_sign = false;
    bool rule_police_gesture = false;
    bool rule_unprotected_left_yield = false;
    bool rule_pedestrian_aeb = false;
    bool rule_bicycle_follow = false;
    bool rule_lead_vehicle_idm = false;
    bool rule_deadlock_breaker = false;
    bool rule_overtaking_active = false;
    bool rule_waiting_to_overtake = false;
    bool crash_prevention_active = false;
    bool autopark_active = false;
    bool reverse_maneuver_active = false;

    const char* primary_reason = "CRUISE_PURE_PURSUIT";

    Vec2 lookahead_pt{};
    float lookahead_dist = 5.0f;
    float cross_track_error = 0.0f;
    float heading_error = 0.0f;
    float target_speed = 10.5f;
    float corner_speed_limit = 12.0f;
    float dist_to_stop_line = 99.0f;
    float dilemma_min_stop_dist = 0.0f;
    float dilemma_max_clear_dist = 0.0f;
    float evasive_lat_offset = 0.0f;

    int visible_objects_360 = 0;
    int visible_free_parking_360 = 0;
    int detected_pedestrian_id = -1;
    int detected_bicycle_id = -1;
    int detected_lead_car_id = -1;
    int detected_parked_car_id = -1;
    int conflict_oncoming_car_id = -1;
    int detected_crash_car_id = -1;
    int target_parking_space_id = -1;

    float expert_steer_cmd = 0.0f;
    float expert_accel_cmd = 0.0f;
};

class RuleBasedDriverV11 {
public:
    RuleTelemetry telemetry;
    std::vector<PathSample> planned_ribbon;
    float overtake_hold_timer = 0.0f;
    const char* active_overtake_reason = "OVERTAKE_OBSTACLE";

    // Autopark & Reverse K-Turn State Machine
    AutoparkState autopark_state = AutoparkState::INACTIVE;
    int locked_parking_space_id = -1;
    int maneuver_subphase = 0;
    int straighten_attempts = 0;
    float maneuver_timer = 0.0f;
    float k_turn_reverse_timer = 0.0f;
    float k_turn_cooldown_timer = 0.0f;
    Vec2 maneuver_start_pos{};
    float maneuver_start_heading = 0.0f;

    // UI Button / Key [P] Handler:
    //   - If INACTIVE: starts searching 360-degree LOS for the closest vacant parking space
    //   - If SEARCHING_360_LOS (not found yet): cancels the search immediately and continues driving
    //   - If PARKED or in the process of parking (APPROACHING / SETUP / MANEUVER): initiates collision-free UNPARKING_EXIT
    void toggle_autopark_button(CityWorld& world) {
        if (autopark_state == AutoparkState::INACTIVE) {
            autopark_state = AutoparkState::SEARCHING_360_LOS;
            locked_parking_space_id = -1;
            maneuver_subphase = 0;
            maneuver_timer = 0.0f;
            world.ego.gear = GearState::DRIVE;
        } else if (autopark_state == AutoparkState::SEARCHING_360_LOS) {
            // Search cancelled before locking a space -> resume normal driving immediately
            autopark_state = AutoparkState::INACTIVE;
            locked_parking_space_id = -1;
            world.ego.gear = GearState::DRIVE;
        } else if (autopark_state == AutoparkState::APPROACHING_AISLE ||
                   autopark_state == AutoparkState::SETUP_POSITION ||
                   autopark_state == AutoparkState::MANEUVER_INTO_STALL ||
                   autopark_state == AutoparkState::PARKED_IN_BOUNDS) {
            // If still on the main road during early approach, resume driving directly;
            // otherwise execute full collision-free UNPARKING_EXIT from the stall/lot
            const ParkingSpace* sp = find_locked_space(world);
            if (!sp || (autopark_state == AutoparkState::APPROACHING_AISLE &&
                        (world.ego.pos - sp->center).len() > 11.5f)) {
                autopark_state = AutoparkState::INACTIVE;
                locked_parking_space_id = -1;
                world.ego.gear = GearState::DRIVE;
            } else {
                autopark_state = AutoparkState::UNPARKING_EXIT;
                maneuver_subphase = 0;
                maneuver_timer = 0.0f;
            }
        }
    }

    const ParkingSpace* find_locked_space(const CityWorld& world) const {
        if (locked_parking_space_id < 0) return nullptr;
        for (const auto& sp : world.parking_spaces) {
            if (sp.id == locked_parking_space_id) return &sp;
        }
        return nullptr;
    }

    // Check 360-degree proximity along motion_dir (+fwd for DRIVE, -fwd for REVERSE) against visible moving NPC cars
    // and non-stall road hazards so the car never touches another vehicle without false-triggering on adjacent stalls
    bool has_close_obstacle_in_motion_dir(const CityWorld& world, Vec2 motion_dir, float max_clear_d,
                                          int ignore_space_id = -1) const {
        const Vehicle& ego = world.ego;
        Vec2 m_left = motion_dir.perp_left();
        for (const auto& npc : world.npc_cars) {
            if (!npc.is_visible_to_ego) continue;
            Vec2 rel = npc.pos - ego.pos;
            float lon = rel.dot(motion_dir);
            float lat = std::fabs(rel.dot(m_left));
            if (lon > 1.0f && lon < max_clear_d && lat < 1.95f) return true;
        }
        for (const auto& pk : world.parked_cars) {
            if (!pk.is_visible_to_ego) continue;
            // Stationary cars parked inside their own marked stalls (pk.parking_space_id >= 0) stay inside their stall bounds;
            // only check tight physical bumper clearance so adjacent stalls never freeze Ego mid-turn
            if (ignore_space_id >= 0 && pk.parking_space_id == ignore_space_id) continue;
            Vec2 rel = pk.pos - ego.pos;
            float lon = rel.dot(motion_dir);
            float lat = std::fabs(rel.dot(m_left));
            if (pk.parking_space_id < 0) {
                if (lon > 1.1f && lon < 3.2f && lat < 1.35f) return true;
            } else {
                if (lon > 0.8f && lon < 2.25f && lat < 0.82f) return true;
            }
        }
        return false;
    }

    // Check if the adjacent passing lane (Lane 2) is clear of moving traffic before pulling out for lateral parking
    bool is_lane2_clear_for_lateral_approach(const CityWorld& world, const ParkingSpace& sp) const {
        const Vehicle& ego = world.ego;
        Vec2 road_fwd = Vec2::from_angle(sp.aisle_heading);
        Vec2 road_left = road_fwd.perp_left();
        float ego_lat_from_stall = (ego.pos - sp.center).dot(road_left);

        // If Ego is already centered in Lane 2 (ego_lat_from_stall > 1.8m), only check ahead along Lane 2
        for (const auto& npc : world.npc_cars) {
            if (npc.is_crashed || npc.is_stopped_at_bus_zone) continue;
            if (!world.has_360_overtake_safety_los(npc.pos)) continue;

            Vec2 rel = npc.pos - ego.pos;
            float lon = rel.dot(road_fwd);
            float lat_from_stall = (npc.pos - sp.center).dot(road_left);
            Vec2 nfwd = Vec2::from_angle(npc.heading);

            // Moving car in Lane 2 (lat_from_stall in [1.1m .. 3.9m] left of the lateral stall row)
            if (lat_from_stall > 1.05f && lat_from_stall < 3.90f) {
                if (nfwd.dot(road_fwd) < -0.30f && lon > -2.0f && lon < 28.0f) return false;
                if (ego_lat_from_stall < 1.65f && nfwd.dot(road_fwd) > 0.30f && lon > -12.0f && lon < 14.0f) return false;
            }
        }
        return true;
    }

    // Smart Decision Helper: For a lateral parking space on the first lane (sp.is_on_lane_one == true),
    // check if the direct forward path along Lane 1 into sp.center is clear of any upstream parked car.
    // When clear, Ego can drive directly straight forward into the spot on Lane 1 instead of shifting to Lane 2 and reversing
    bool can_enter_lateral_space_directly_forward(const CityWorld& world, const ParkingSpace& sp) const {
        if (sp.layout != ParkingLayoutType::PARALLEL_LATERAL_REVERSE || !sp.is_on_lane_one) return false;
        const Vehicle& ego = world.ego;
        Vec2 s_fwd = Vec2::from_angle(sp.heading);
        Vec2 s_left = s_fwd.perp_left();
        float lon_to_center = (sp.center - ego.pos).dot(s_fwd);
        float lat_from_stall = std::fabs((ego.pos - sp.center).dot(s_left));

        // Must be upstream of the stall (or actively pulling straight into it) and not already parallel-setup past it in Lane 2
        if (lon_to_center < -0.25f) return false;
        if (lat_from_stall > 1.40f && lon_to_center < 5.5f) return false;

        // Check if any visible stationary parked car sits on Lane 1 between ego.pos and sp.center
        for (const auto& pk : world.parked_cars) {
            if (!pk.is_visible_to_ego || pk.parking_space_id == sp.id) continue;
            float pk_lon_from_ego = (pk.pos - ego.pos).dot(s_fwd);
            float pk_lon_to_stall = (sp.center - pk.pos).dot(s_fwd);
            float pk_lat = std::fabs((pk.pos - sp.center).dot(s_left));
            if (pk_lat < 1.45f && pk_lon_from_ego > -1.5f && pk_lon_to_stall > 1.0f) {
                return false; // Upstream parked car on Lane 1 blocks direct straight-in entry
            }
        }
        return true;
    }

    // Check if there is a visible parked car immediately ahead of Ego in the same lateral row (requiring reverse/Lane-2 exit)
    bool has_parked_car_blocking_forward_lateral_exit(const CityWorld& world, const ParkingSpace& sp, float check_dist = 8.0f) const {
        Vec2 s_fwd = Vec2::from_angle(sp.heading);
        Vec2 s_left = s_fwd.perp_left();
        for (const auto& pk : world.parked_cars) {
            if (!pk.is_visible_to_ego || pk.parking_space_id == sp.id) continue;
            float lon = (pk.pos - world.ego.pos).dot(s_fwd);
            float lat = std::fabs((pk.pos - sp.center).dot(s_left));
            if (lon > 0.5f && lon < check_dist && lat < 1.45f) {
                return true;
            }
        }
        return false;
    }

    // Precision Straight Stall-Centerline Axis Tracker (Forward & Reverse â€” guarantees 100% straight, centered alignment)
    void track_stall_centerline_axis(const CityWorld& world, const ParkingSpace& sp,
                                     float lon_offset_along_heading, GearState desired_gear, float max_v) {
        const Vehicle& ego = world.ego;
        Vec2 s_fwd = Vec2::from_angle(sp.heading);
        Vec2 s_left = s_fwd.perp_left();
        Vec2 target_pt = sp.center + s_fwd * lon_offset_along_heading;
        Vec2 motion_axis = (desired_gear == GearState::REVERSE) ? (s_fwd * -1.0f) : s_fwd;
        float dir_sign = (desired_gear == GearState::REVERSE) ? -1.0f : 1.0f;

        Vec2 ego_fwd = Vec2::from_angle(ego.heading);
        Vec2 ego_motion = (desired_gear == GearState::REVERSE) ? (ego_fwd * -1.0f) : ego_fwd;
        if (has_close_obstacle_in_motion_dir(world, ego_motion, 3.8f, locked_parking_space_id)) {
            telemetry.target_speed = 0.0f;
            telemetry.expert_steer_cmd = 0.0f;
            telemetry.expert_accel_cmd = -0.95f;
            return;
        }

        // Visual ribbon along the stall centerline
        build_autopark_visual_ribbon(ego, (ego.pos + target_pt) * 0.5f, target_pt, sp.heading);

        // Signed lateral offset from the stall centerline (positive when stall centerline is to the car's +s_left)
        float e_lat = (sp.center - ego.pos).dot(s_left);
        float h_des = wrap_angle(sp.heading + dir_sign * std::atan(0.95f * e_lat));
        float steer_rad = -dir_sign * 1.65f * wrap_angle(ego.heading - h_des);

        telemetry.expert_steer_cmd = clampf(steer_rad / 0.68f, -1.0f, 1.0f);
        telemetry.lookahead_pt = target_pt;

        float lon_rem = std::max(0.0f, (target_pt - ego.pos).dot(motion_axis));
        telemetry.lookahead_dist = lon_rem;

        float v_des = (lon_rem < 0.05f) ? 0.0f : clampf(std::sqrt(2.0f * 1.15f * lon_rem), 0.32f, max_v);
        telemetry.target_speed = v_des;

        float v_err = v_des - ego.speed;
        if (v_des < 0.06f && ego.speed < 0.35f) {
            telemetry.expert_accel_cmd = -0.95f;
        } else if (v_err >= 0.0f) {
            telemetry.expert_accel_cmd = clampf(v_err * 0.55f, 0.0f, 0.65f);
        } else {
            telemetry.expert_accel_cmd = clampf(v_err * 0.85f, -1.0f, 0.0f);
        }
    }

    // Fluid Continuous Cubic-Bezier Spline Path Follower for Autopark (Forward & Reverse â€” Zero Snapping)
    void track_parking_bezier_curve(const CityWorld& world, const CubicBezier2D& curve,
                                    float final_heading, GearState desired_gear, float max_v) {
        const Vehicle& ego = world.ego;
        Vec2 fwd = Vec2::from_angle(ego.heading);
        Vec2 motion_dir = (desired_gear == GearState::REVERSE) ? (fwd * -1.0f) : fwd;
        Vec2 final_motion_dir = (desired_gear == GearState::REVERSE)
                                ? (Vec2::from_angle(final_heading) * -1.0f)
                                : Vec2::from_angle(final_heading);
        float dir_sign = (desired_gear == GearState::REVERSE) ? -1.0f : 1.0f;

        if (has_close_obstacle_in_motion_dir(world, motion_dir, 4.2f, locked_parking_space_id)) {
            telemetry.target_speed = 0.0f;
            telemetry.expert_steer_cmd = 0.0f;
            telemetry.expert_accel_cmd = -0.95f;
            return;
        }

        // Sample the Cubic Bezier curve at 36 points, populate planned_ribbon, and find closest t_best
        planned_ribbon.clear();
        constexpr int N_SAMPLES = 36;
        float best_d2 = 1e9f;
        int best_idx = 0;
        float cum_s = 0.0f;
        Vec2 prev_p = curve.eval(0.0f);

        for (int i = 0; i <= N_SAMPLES; ++i) {
            float t = static_cast<float>(i) / N_SAMPLES;
            Vec2 p = curve.eval(t);
            if (i > 0) cum_s += (p - prev_p).len();
            prev_p = p;

            Vec2 tan = curve.tangent(t);
            float th = (tan.len_sq() > 1e-5f)
                       ? std::atan2(tan.y, tan.x)
                       : ((desired_gear == GearState::REVERSE) ? wrap_angle(final_heading + PI) : final_heading);
            // Note: in REVERSE, the curve tangent points in the direction of backing travel (-fwd_car),
            // so the desired car nose heading along the curve is wrap_angle(th + PI)
            float car_h = (desired_gear == GearState::REVERSE) ? wrap_angle(th + PI) : th;

            PathSample ps;
            ps.pos = p;
            ps.heading = car_h;
            ps.s = cum_s;
            planned_ribbon.push_back(ps);

            float d2 = (p - ego.pos).len_sq();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = i;
            }
        }
        planned_ribbon.back().heading = final_heading;

        float total_s = planned_ribbon.back().s;
        float cur_s   = planned_ribbon[best_idx].s;
        float rem_s   = std::max(0.0f, total_s - cur_s);
        float dist_to_end = (curve.p3 - ego.pos).len();

        // Pick reference point 0.45m ahead along the spline (or extended along the final stall axis past curve.p3
        // so the tracker NEVER twists the steering wheel near the end of the curve)
        float look_s = cur_s + 0.45f;
        Vec2 ref_pos = curve.p3;
        float ref_car_h = final_heading;
        int ref_idx = N_SAMPLES;

        if (look_s <= total_s) {
            for (int i = best_idx; i <= N_SAMPLES; ++i) {
                if (planned_ribbon[i].s >= look_s) {
                    ref_pos   = planned_ribbon[i].pos;
                    ref_car_h = planned_ribbon[i].heading;
                    ref_idx   = i;
                    break;
                }
            }
        } else {
            float overshoot_s = look_s - total_s;
            ref_pos   = curve.p3 + final_motion_dir * overshoot_s;
            ref_car_h = final_heading;
            ref_idx   = N_SAMPLES;
        }

        // Exact Ackermann curvature feedforward d(car_h)/ds + nonlinear Stanley cross-track & heading feedback
        int i0 = std::max(0, ref_idx - 1);
        int i1 = std::min(N_SAMPLES, ref_idx + 1);
        float ds_seg = std::max(0.15f, planned_ribbon[i1].s - planned_ribbon[i0].s);
        float d_car_h_ds = wrap_angle(planned_ribbon[i1].heading - planned_ribbon[i0].heading) / ds_seg;
        if (rem_s < 0.70f) {
            d_car_h_ds *= (rem_s / 0.70f);
        }

        Vec2 ref_fwd = Vec2::from_angle(ref_car_h);
        Vec2 ref_left = ref_fwd.perp_left();
        float e_lat = (ref_pos - ego.pos).dot(ref_left);

        float h_des = wrap_angle(ref_car_h + dir_sign * std::atan(0.85f * e_lat));
        float ff_steer = std::atan(dir_sign * ego.wheelbase * d_car_h_ds);
        float fb_steer = -dir_sign * 1.60f * wrap_angle(ego.heading - h_des);
        float steer_rad = ff_steer + fb_steer;

        telemetry.expert_steer_cmd = clampf(steer_rad / 0.68f, -1.0f, 1.0f);
        telemetry.lookahead_pt = ref_pos;
        telemetry.lookahead_dist = dist_to_end;

        // Smooth kinematic deceleration profile down to 0.0 m/s at curve.p3 (using max(rem_s, lon_rem) so lateral offset never stops early)
        float lon_rem = std::max(0.0f, (curve.p3 - ego.pos).dot(final_motion_dir));
        float decel_d = std::max(rem_s, lon_rem);
        float v_des = (decel_d < 0.06f) ? 0.0f : clampf(std::sqrt(2.0f * 1.15f * decel_d), 0.34f, max_v);
        telemetry.target_speed = v_des;

        float v_err = v_des - ego.speed;
        if (v_des < 0.06f && ego.speed < 0.35f) {
            telemetry.expert_accel_cmd = -0.95f;
        } else if (v_err >= 0.0f) {
            telemetry.expert_accel_cmd = clampf(v_err * 0.55f, 0.0f, 0.68f);
        } else {
            telemetry.expert_accel_cmd = clampf(v_err * 0.82f, -1.0f, 0.0f);
        }
    }

    // Construct a world-fixed Cubic Bezier curve from a recorded start pose (start_pos, start_heading)
    // to a target pose (target_pt, target_heading) so the reference spline never shifts with vehicle drift
    static CubicBezier2D build_fixed_pose_bezier(Vec2 start_pos, float start_heading,
                                                 Vec2 target_pt, float target_heading,
                                                 GearState desired_gear, bool align_heading_strictly = true) {
        Vec2 d_in = (desired_gear == GearState::REVERSE)
                    ? (Vec2::from_angle(start_heading) * -1.0f)
                    : Vec2::from_angle(start_heading);
        Vec2 d_out = (desired_gear == GearState::REVERSE)
                     ? (Vec2::from_angle(target_heading) * -1.0f)
                     : Vec2::from_angle(target_heading);
        float chord = (target_pt - start_pos).len();
        float c_in  = clampf(chord * 0.36f, 0.5f, 2.5f);
        float c_out = align_heading_strictly ? clampf(chord * 0.44f, 1.2f, 3.1f) : c_in;
        return {
            start_pos,
            start_pos + d_in * c_in,
            target_pt - d_out * c_out,
            target_pt
        };
    }

    // Dynamic Point-to-Point Pose Controller (used during unpark exit using Pure-Pursuit + Heading alignment)
    void compute_parking_control(const CityWorld& world, Vec2 target_pt, float target_heading,
                                 GearState desired_gear, float max_v, bool align_heading_strictly) {
        const Vehicle& ego = world.ego;
        Vec2 fwd = Vec2::from_angle(ego.heading);
        Vec2 left = fwd.perp_left();
        Vec2 motion_dir = (desired_gear == GearState::REVERSE) ? (fwd * -1.0f) : fwd;
        float dir_sign = (desired_gear == GearState::REVERSE) ? -1.0f : 1.0f;

        if (has_close_obstacle_in_motion_dir(world, motion_dir, 4.2f, locked_parking_space_id)) {
            telemetry.target_speed = 0.0f;
            telemetry.expert_steer_cmd = 0.0f;
            telemetry.expert_accel_cmd = -0.95f;
            return;
        }

        Vec2 to_t = target_pt - ego.pos;
        float dist = to_t.len();
        Vec2 d_out = Vec2::from_angle(target_heading);
        // Lead point shifts along target_heading centerline as we get closer so we align heading smoothly
        float lead_back = align_heading_strictly ? clampf(dist * 0.45f, 0.0f, 1.8f) : 0.0f;
        Vec2 aim_pt = target_pt - d_out * (dir_sign * lead_back);
        Vec2 to_aim = aim_pt - ego.pos;
        float ld = std::max(0.75f, to_aim.len());
        float lat_y = to_aim.dot(left);
        float h_err = wrap_angle(ego.heading - target_heading);

        float pp = std::atan(2.0f * ego.wheelbase * lat_y / (ld * ld));
        float steer_rad = pp - dir_sign * 0.65f * h_err;

        telemetry.expert_steer_cmd = clampf(steer_rad / 0.68f, -1.0f, 1.0f);
        telemetry.lookahead_pt = aim_pt;
        telemetry.lookahead_dist = dist;

        float v_des = (dist < 0.08f) ? 0.0f : clampf(std::sqrt(2.0f * 1.15f * dist), 0.35f, max_v);
        telemetry.target_speed = v_des;
        float v_err = v_des - ego.speed;
        if (v_des < 0.08f && ego.speed < 0.35f) {
            telemetry.expert_accel_cmd = -0.95f;
        } else if (v_err >= 0.0f) {
            telemetry.expert_accel_cmd = clampf(v_err * 0.55f, 0.0f, 0.68f);
        } else {
            telemetry.expert_accel_cmd = clampf(v_err * 0.82f, -1.0f, 0.0f);
        }
    }

    // Build a smooth visual ribbon for the 3D trajectory overlay during parking maneuvers
    void build_autopark_visual_ribbon(const Vehicle& ego, Vec2 p_mid, Vec2 p_end, float end_heading) {
        planned_ribbon.clear();
        CubicBezier2D bez = {
            ego.pos,
            ego.pos * 0.55f + p_mid * 0.45f,
            p_mid * 0.55f + p_end * 0.45f,
            p_end
        };
        float cum_s = 0.0f;
        Vec2 prev = ego.pos;
        for (int i = 0; i <= 20; ++i) {
            float t = static_cast<float>(i) / 20.0f;
            Vec2 p = bez.eval(t);
            if (i > 0) cum_s += (p - prev).len();
            prev = p;
            PathSample ps;
            ps.pos = p;
            ps.heading = end_heading;
            ps.s = cum_s;
            planned_ribbon.push_back(ps);
        }
    }

    // Execute active Autopark / Unpark state machine; returns true if Autopark is controlling Ego this frame
    bool step_autopark_state_machine(CityWorld& world, float dt, bool apply_controls_to_ego) {
        Vehicle& ego = world.ego;
        const LaneEdge& cur_edge = world.edges[ego.route_edges[ego.current_route_idx]];
        Vec2 fwd = Vec2::from_angle(ego.heading);

        if (autopark_state == AutoparkState::INACTIVE) {
            return false;
        }

        // State 1: Scanning 360-degree Line-of-Sight for the closest vacant rectangular parking space
        if (autopark_state == AutoparkState::SEARCHING_360_LOS) {
            telemetry.autopark_active = true;
            const ParkingSpace* best_sp = nullptr;
            float best_dist = 1e9f;

            for (const auto& sp : world.parking_spaces) {
                // Strictly only what the car sees in 360-degree LOS (zero map god-view) and only vacant stalls
                if (!sp.is_visible_to_ego || sp.is_occupied) continue;

                if (sp.road_edge_id == cur_edge.id) {
                    Vec2 road_d = Vec2::from_angle(sp.aisle_heading);
                    float ds_turn_in = (sp.road_turn_in_pt - ego.pos).dot(road_d);
                    float ds_entry   = (sp.aisle_entry_pt - ego.pos).dot(road_d);
                    float ds_center  = (sp.center - ego.pos).dot(road_d);
                    float d_euclid   = (sp.center - ego.pos).len();
                    bool direct_lane1_ok = can_enter_lateral_space_directly_forward(world, sp) && (ds_center > 2.8f);
                    if (((ds_turn_in > -3.5f && ds_entry > -1.5f) || direct_lane1_ok) && d_euclid < best_dist) {
                        best_dist = d_euclid;
                        best_sp = &sp;
                    }
                }
            }

            if (best_sp != nullptr) {
                locked_parking_space_id = best_sp->id;
                autopark_state = AutoparkState::APPROACHING_AISLE;
                maneuver_subphase = 0;
                maneuver_timer = 0.0f;
            } else {
                // If a visible vacant space is spotted in 360 LOS on an upcoming connected street,
                // bias Ego's upcoming turn at the next intersection toward that street
                for (const auto& sp : world.parking_spaces) {
                    if (!sp.is_visible_to_ego || sp.is_occupied) continue;
                    for (int conn_id : cur_edge.next_edges) {
                        const auto& conn = world.edges[conn_id];
                        if (!conn.next_edges.empty() && conn.next_edges[0] == sp.road_edge_id) {
                            ego.route_edges.resize(ego.current_route_idx + 1);
                            ego.route_edges.push_back(conn_id);
                            ego.route_edges.push_back(sp.road_edge_id);
                            world.ensure_route_horizon(ego);
                            break;
                        }
                    }
                }
                return false; // Continue normal road driving while scanning 360 LOS
            }
        }

        const ParkingSpace* sp = find_locked_space(world);
        if (!sp) {
            autopark_state = AutoparkState::INACTIVE;
            ego.gear = GearState::DRIVE;
            return false;
        }

        telemetry.autopark_active = true;
        telemetry.target_parking_space_id = sp->id;
        maneuver_timer += dt;

        // State 2: Approaching Rectangular Parking Lot Driveway / Lateral Approach Lane
        if (autopark_state == AutoparkState::APPROACHING_AISLE) {
            ego.gear = GearState::DRIVE;
            Vec2 aisle_d = Vec2::from_angle(sp->aisle_heading);
            float dist_to_turn_in = (sp->road_turn_in_pt - ego.pos).dot(aisle_d);
            float dist_to_center  = (sp->center - ego.pos).dot(aisle_d);

            // Smart Decision: If this is a Lane-1 lateral parking space and the direct Lane-1 path into the spot is open,
            // do NOT shift to Lane 2 and reverse â€” stay on Lane 1 and glide directly straight forward into the spot
            if (can_enter_lateral_space_directly_forward(world, *sp)) {
                if (cur_edge.is_intersection_connector || dist_to_center > 7.5f) {
                    maneuver_timer = 0.0f;
                    return false; // Stay on Lane 1 using normal road lane-keeping until 7.5m before the spot
                }
                autopark_state = AutoparkState::MANEUVER_INTO_STALL;
                maneuver_subphase = 3; // Direct straight-in forward entry on Lane 1
                maneuver_timer = 0.0f;
                maneuver_start_pos = ego.pos;
                maneuver_start_heading = ego.heading;
                return step_autopark_state_machine(world, 0.0f, apply_controls_to_ego);
            }

            // CRITICAL: Never leave the road early or cut diagonally across the sidewalk/grass
            // Stay 100% inside the road lane using normal road lane-keeping until reaching sp->road_turn_in_pt
            // right at the flush driveway opening (or lateral parking row entrance).
            if (maneuver_subphase == 0 && (cur_edge.is_intersection_connector || dist_to_turn_in > 0.8f)) {
                maneuver_timer = 0.0f;
                return false;
            }

            // Record the fixed start pose at the driveway turn-in point once so the driveway spline never drifts
            if (maneuver_subphase == 0) {
                maneuver_start_pos = ego.pos;
                maneuver_start_heading = ego.heading;
                maneuver_subphase = 1;
                maneuver_timer = 0.0f;
            }

            // Before pulling into Lane 2 for Lane-1 lateral parking, check that Lane 2 is clear of moving cars
            if (sp->layout == ParkingLayoutType::PARALLEL_LATERAL_REVERSE &&
                sp->is_on_lane_one &&
                !is_lane2_clear_for_lateral_approach(world, *sp)) {
                telemetry.primary_reason = "AUTOPARK_WAIT_LANE2_CLEAR";
                telemetry.target_speed = 0.0f;
                telemetry.expert_steer_cmd = 0.0f;
                telemetry.expert_accel_cmd = -0.90f;
                maneuver_timer = 0.0f;
            } else {
                telemetry.primary_reason = (sp->lot_id > 0) ? "AUTOPARK_ENTER_LOT_AISLE" : "AUTOPARK_APPROACH_LATERAL";
                CubicBezier2D entry_curve = build_fixed_pose_bezier(
                    maneuver_start_pos, maneuver_start_heading, sp->aisle_entry_pt, sp->aisle_heading, GearState::DRIVE, true);
                track_parking_bezier_curve(world, entry_curve, sp->aisle_heading, GearState::DRIVE, 3.2f);

                float lon_past_entry = (ego.pos - sp->aisle_entry_pt).dot(aisle_d);
                float lat_from_entry = std::fabs((ego.pos - sp->aisle_entry_pt).dot(aisle_d.perp_left()));
                if ((ego.pos - sp->aisle_entry_pt).len() < 0.65f ||
                    (lon_past_entry > -0.10f && lat_from_entry < 0.85f) ||
                    maneuver_timer > 6.5f) {
                    autopark_state = AutoparkState::SETUP_POSITION;
                    maneuver_subphase = 0;
                    maneuver_timer = 0.0f;
                    maneuver_start_pos = ego.pos;
                    maneuver_start_heading = ego.heading;
                }
            }
        }
        // State 3: Positioning at Setup Point in Lot Aisle / Passing Lane before Final Stall Entry
        else if (autopark_state == AutoparkState::SETUP_POSITION) {
            ego.gear = GearState::DRIVE;
            telemetry.primary_reason = (sp->layout == ParkingLayoutType::PARALLEL_LATERAL_REVERSE)
                                       ? "AUTOPARK_PARALLEL_SETUP" : "AUTOPARK_AISLE_SETUP";
            CubicBezier2D setup_curve = build_fixed_pose_bezier(
                maneuver_start_pos, maneuver_start_heading, sp->setup_pt, sp->setup_heading, GearState::DRIVE, true);
            track_parking_bezier_curve(world, setup_curve, sp->setup_heading, GearState::DRIVE, 2.4f);

            Vec2 setup_d = Vec2::from_angle(sp->setup_heading);
            float lon_past_setup = (ego.pos - sp->setup_pt).dot(setup_d);
            float lat_from_setup = std::fabs((ego.pos - sp->setup_pt).dot(setup_d.perp_left()));
            if ((ego.pos - sp->setup_pt).len() < 0.32f ||
                (lon_past_setup >= -0.06f && lat_from_setup < 0.45f && ego.speed < 0.45f) ||
                maneuver_timer > 5.8f) {
                autopark_state = AutoparkState::MANEUVER_INTO_STALL;
                maneuver_subphase = 0;
                straighten_attempts = 0;
                maneuver_timer = 0.0f;
                // Record the fixed start pose of the final stall entry spline so the Bezier curve never shifts
                maneuver_start_pos = ego.pos;
                maneuver_start_heading = ego.heading;
            }
        }
        // State 4: Fluid Continuous Bezier-Spline Stall Entry + Centerline Straightening Pass (Zero Snapping, Zero Crookedness)
        else if (autopark_state == AutoparkState::MANEUVER_INTO_STALL) {
            Vec2 s_fwd = Vec2::from_angle(sp->heading);
            Vec2 s_left = s_fwd.perp_left();
            float e_lat_abs = std::fabs((sp->center - ego.pos).dot(s_left));
            float h_err_abs = std::fabs(wrap_angle(ego.heading - sp->heading));

            if (sp->layout == ParkingLayoutType::PARALLEL_LATERAL_REVERSE) {
                if (maneuver_subphase == 3) {
                    // Subphase 3: Direct Straight-In Forward Entry on Lane 1 (no unnecessary detour to Lane 2 or reverse)
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = "AUTOPARK_LANE1_DIRECT_STRAIGHT_IN";
                    CubicBezier2D direct_curve = build_fixed_pose_bezier(
                        maneuver_start_pos, maneuver_start_heading, sp->center, sp->heading, GearState::DRIVE, true);
                    track_parking_bezier_curve(world, direct_curve, sp->heading, GearState::DRIVE, 2.25f);

                    float rem_fwd = (sp->center - ego.pos).dot(s_fwd);
                    if (rem_fwd <= 0.06f || (ego.pos - sp->center).len() < 0.14f || maneuver_timer > 5.5f) {
                        autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                        ego.gear = GearState::PARK;
                        ego.speed = 0.0f;
                    }
                } else if (maneuver_subphase == 0) {
                    // Subphase 0: 6.0m Reverse S-Curve Cubic Bezier from setup lane into the rear portion of the lateral stall
                    ego.gear = GearState::REVERSE;
                    telemetry.reverse_maneuver_active = true;
                    telemetry.primary_reason = sp->is_on_lane_one ? "AUTOPARK_LANE1_PARALLEL_REV" : "AUTOPARK_CURB_PARALLEL_REV";

                    Vec2 rev_target = sp->center - s_fwd * 0.90f - s_left * 0.14f;
                    float half_lon = std::max(2.50f, std::fabs((maneuver_start_pos - rev_target).dot(s_fwd)) * 0.50f);
                    CubicBezier2D s_curve = {
                        maneuver_start_pos,
                        maneuver_start_pos - s_fwd * half_lon,
                        rev_target + s_fwd * half_lon,
                        rev_target
                    };
                    track_parking_bezier_curve(world, s_curve, sp->heading, GearState::REVERSE, 1.70f);

                    float x_lon = (ego.pos - rev_target).dot(s_fwd);
                    if ((ego.pos - rev_target).len() < 0.24f || (x_lon <= 0.10f && ego.speed < 0.38f) || maneuver_timer > 6.8f) {
                        // Subphase 1: Pull forward 0.90m inside the parallel stall along the centerline to straighten 100% parallel to the curb
                        maneuver_subphase = 1;
                        maneuver_timer = 0.0f;
                    }
                } else if (maneuver_subphase == 1) {
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = "AUTOPARK_PARALLEL_STRAIGHTEN_FWD";
                    track_stall_centerline_axis(world, *sp, 0.0f, GearState::DRIVE, 1.15f);

                    float rem_fwd = (sp->center - ego.pos).dot(s_fwd);
                    if (rem_fwd <= 0.06f || (ego.pos - sp->center).len() < 0.14f || maneuver_timer > 3.2f) {
                        autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                        ego.gear = GearState::PARK;
                        ego.speed = 0.0f;
                    }
                }
            } else if (sp->layout == ParkingLayoutType::PERPENDICULAR_90_REVERSE) {
                Vec2 out_dir = s_fwd;
                Vec2 in_dir  = out_dir * -1.0f;
                if (maneuver_subphase == 0) {
                    // Reverse 90-Degree Cubic Bezier from maneuver_start_pos (in Lot Aisle) into sp->center
                    ego.gear = GearState::REVERSE;
                    telemetry.reverse_maneuver_active = true;
                    telemetry.primary_reason = "AUTOPARK_90_BAY_REVERSE_IN";

                    Vec2 start_back_dir = Vec2::from_angle(maneuver_start_heading) * -1.0f;
                    CubicBezier2D rev_bay_curve = {
                        maneuver_start_pos,
                        maneuver_start_pos + start_back_dir * 2.25f,
                        sp->center - in_dir * 3.30f,
                        sp->center
                    };
                    track_parking_bezier_curve(world, rev_bay_curve, sp->heading, GearState::REVERSE, 1.75f);

                    float rem_back = (sp->center - ego.pos).dot(in_dir);
                    if ((ego.pos - sp->center).len() < 0.22f || (rem_back <= 0.10f && ego.speed < 0.38f) || maneuver_timer > 7.2f) {
                        if (e_lat_abs < 0.10f && h_err_abs < 0.035f) {
                            autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                            ego.gear = GearState::PARK;
                            ego.speed = 0.0f;
                        } else {
                            // Pull forward 2.35m onto stall centerline using a fixed Bezier to eliminate any crookedness
                            maneuver_subphase = 1;
                            maneuver_timer = 0.0f;
                            maneuver_start_pos = ego.pos;
                            maneuver_start_heading = ego.heading;
                        }
                    }
                } else if (maneuver_subphase == 1) {
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = "AUTOPARK_90_PULL_FWD_STRAIGHTEN";
                    Vec2 pull_pt = sp->center + out_dir * 2.35f;
                    CubicBezier2D pull_curve = build_fixed_pose_bezier(
                        maneuver_start_pos, maneuver_start_heading, pull_pt, sp->heading, GearState::DRIVE, true);
                    track_parking_bezier_curve(world, pull_curve, sp->heading, GearState::DRIVE, 1.45f);

                    float rem_pull = (pull_pt - ego.pos).dot(out_dir);
                    if (rem_pull <= 0.10f || (ego.pos - pull_pt).len() < 0.22f || maneuver_timer > 3.2f) {
                        maneuver_subphase = 2;
                        maneuver_timer = 0.0f;
                    }
                } else {
                    ego.gear = GearState::REVERSE;
                    telemetry.reverse_maneuver_active = true;
                    telemetry.primary_reason = "AUTOPARK_90_FINAL_STRAIGHT_REV";
                    track_stall_centerline_axis(world, *sp, 0.0f, GearState::REVERSE, 1.30f);

                    float rem_back = (sp->center - ego.pos).dot(in_dir);
                    if (rem_back <= 0.06f || (ego.pos - sp->center).len() < 0.16f || maneuver_timer > 3.5f) {
                        autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                        ego.gear = GearState::PARK;
                        ego.speed = 0.0f;
                    }
                }
            } else {
                // PERPENDICULAR_90_FRONT or ANGLED_45_FRONT: Forward Cubic Bezier + Reverse/Forward Straightening if needed
                Vec2 in_dir = s_fwd;
                if (maneuver_subphase == 0) {
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = (sp->layout == ParkingLayoutType::ANGLED_45_FRONT)
                                               ? "AUTOPARK_45_ANGLED_FRONT_IN" : "AUTOPARK_90_BAY_FRONT_IN";

                    Vec2 start_fwd_dir = Vec2::from_angle(maneuver_start_heading);
                    float lead_in = (sp->layout == ParkingLayoutType::ANGLED_45_FRONT) ? 2.50f : 3.35f;
                    CubicBezier2D fwd_bay_curve = {
                        maneuver_start_pos,
                        maneuver_start_pos + start_fwd_dir * 2.15f,
                        sp->center - in_dir * lead_in,
                        sp->center
                    };
                    track_parking_bezier_curve(world, fwd_bay_curve, sp->heading, GearState::DRIVE, 1.75f);

                    float rem_fwd = (sp->center - ego.pos).dot(in_dir);
                    if ((ego.pos - sp->center).len() < 0.22f || (rem_fwd <= 0.10f && ego.speed < 0.38f) || maneuver_timer > 7.2f) {
                        if (e_lat_abs < 0.10f && h_err_abs < 0.035f) {
                            autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                            ego.gear = GearState::PARK;
                            ego.speed = 0.0f;
                        } else {
                            // Back out 2.35m onto the stall centerline using a fixed Bezier to eliminate any crookedness
                            maneuver_subphase = 1;
                            maneuver_timer = 0.0f;
                            maneuver_start_pos = ego.pos;
                            maneuver_start_heading = ego.heading;
                        }
                    }
                } else if (maneuver_subphase == 1) {
                    ego.gear = GearState::REVERSE;
                    telemetry.reverse_maneuver_active = true;
                    telemetry.primary_reason = "AUTOPARK_BAY_REV_STRAIGHTEN";
                    Vec2 back_pt = sp->center - in_dir * 2.35f;
                    CubicBezier2D back_curve = build_fixed_pose_bezier(
                        maneuver_start_pos, maneuver_start_heading, back_pt, sp->heading, GearState::REVERSE, true);
                    track_parking_bezier_curve(world, back_curve, sp->heading, GearState::REVERSE, 1.45f);

                    float rem_back = (ego.pos - back_pt).dot(in_dir);
                    if (rem_back <= 0.10f || (ego.pos - back_pt).len() < 0.22f || maneuver_timer > 3.2f) {
                        maneuver_subphase = 2;
                        maneuver_timer = 0.0f;
                    }
                } else {
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = "AUTOPARK_BAY_FINAL_STRAIGHT_FWD";
                    track_stall_centerline_axis(world, *sp, 0.0f, GearState::DRIVE, 1.30f);

                    float rem_fwd = (sp->center - ego.pos).dot(in_dir);
                    if (rem_fwd <= 0.06f || (ego.pos - sp->center).len() < 0.16f || maneuver_timer > 3.5f) {
                        autopark_state = AutoparkState::PARKED_IN_BOUNDS;
                        ego.gear = GearState::PARK;
                        ego.speed = 0.0f;
                    }
                }
            }
        }
        // State 5: Parked inside Rectangular Stall Bounds in GearState::PARK (P) â€” ZERO artificial position/heading snapping
        else if (autopark_state == AutoparkState::PARKED_IN_BOUNDS) {
            ego.gear = GearState::PARK;
            ego.speed = 0.0f;
            telemetry.primary_reason = "PARKED_IN_RECTANGULAR_BOUNDS";
            telemetry.target_speed = 0.0f;
            telemetry.expert_steer_cmd = 0.0f;
            telemetry.expert_accel_cmd = -1.0f;
        }
        // State 6: Collision-Free Unparking Exit (Reverse / Pull Out of Stall -> Lot Aisle -> Road Lane)
        else if (autopark_state == AutoparkState::UNPARKING_EXIT) {
            if (maneuver_subphase == 0) {
                if (sp->layout == ParkingLayoutType::PERPENDICULAR_90_FRONT ||
                    sp->layout == ParkingLayoutType::ANGLED_45_FRONT) {
                    // Parked Front-In -> Reverse straight out of the bay first, then swing into lot aisle
                    ego.gear = GearState::REVERSE;
                    telemetry.reverse_maneuver_active = true;
                    telemetry.primary_reason = "UNPARK_REVERSE_OUT_OF_BAY";
                    Vec2 in_dir = Vec2::from_angle(sp->heading);
                    float dist_backed = (sp->center - ego.pos).dot(in_dir);
                    Vec2 rev_target = (dist_backed < 2.1f) ? sp->bay_mouth_pt : sp->setup_pt;
                    float rev_h     = (dist_backed < 2.1f) ? sp->heading      : sp->aisle_heading;
                    compute_parking_control(world, rev_target, rev_h, GearState::REVERSE, 1.80f, dist_backed < 2.1f);
                    build_autopark_visual_ribbon(ego, sp->bay_mouth_pt, sp->setup_pt, sp->aisle_heading);

                    if ((ego.pos - sp->center).len() > 4.1f || (ego.pos - sp->setup_pt).len() < 1.15f || maneuver_timer > 4.2f) {
                        maneuver_subphase = 1;
                        maneuver_timer = 0.0f;
                    }
                } else if (sp->layout == ParkingLayoutType::PERPENDICULAR_90_REVERSE) {
                    // Parked Reverse-In -> Pull straight forward out of the bay to clear adjacent parked cars
                    ego.gear = GearState::DRIVE;
                    telemetry.primary_reason = "UNPARK_DRIVE_OUT_OF_BAY";
                    compute_parking_control(world, sp->bay_mouth_pt, sp->heading, GearState::DRIVE, 1.90f, true);
                    build_autopark_visual_ribbon(ego, sp->bay_mouth_pt, sp->exit_merge_pt, sp->aisle_heading);

                    if ((ego.pos - sp->center).len() > 2.25f || maneuver_timer > 2.2f) {
                        maneuver_subphase = 1;
                        maneuver_timer = 0.0f;
                    }
                } else {
                    // PARALLEL_LATERAL_REVERSE:
                    // Only reverse 0.65m first if a parked car is tight (< 5.2m) in front of Ego in the same lateral row
                    if (has_parked_car_blocking_forward_lateral_exit(world, *sp, 5.2f)) {
                        ego.gear = GearState::REVERSE;
                        telemetry.reverse_maneuver_active = true;
                        telemetry.primary_reason = "UNPARK_PARALLEL_REV_CLEAR";
                        Vec2 back_pt = sp->center - Vec2::from_angle(sp->heading) * 0.65f;
                        compute_parking_control(world, back_pt, sp->heading, GearState::REVERSE, 1.25f, true);
                        if ((ego.pos - back_pt).len() < 0.28f || maneuver_timer > 0.85f) {
                            maneuver_subphase = 1;
                            maneuver_timer = 0.0f;
                        }
                    } else {
                        maneuver_subphase = 1;
                        maneuver_timer = 0.0f;
                    }
                }
            } else {
                // Subphase 1: Drive out along the lot aisle first (so we never cut across the front curb/grass),
                // then steer through the exit driveway opening back onto the road lane
                ego.gear = GearState::DRIVE;
                Vec2 aisle_d = Vec2::from_angle(sp->aisle_heading);
                float dist_to_aisle_exit = (sp->aisle_exit_pt - ego.pos).dot(aisle_d);

                if (sp->lot_id > 0 && dist_to_aisle_exit > 1.2f && maneuver_timer < 3.4f) {
                    telemetry.primary_reason = "UNPARK_AISLE_TO_EXIT_DRIVEWAY";
                    compute_parking_control(world, sp->aisle_exit_pt, sp->aisle_heading, GearState::DRIVE, 3.2f, true);
                    build_autopark_visual_ribbon(ego, sp->aisle_exit_pt, sp->exit_merge_pt, sp->aisle_heading);
                } else {
                    telemetry.primary_reason = "UNPARK_MERGE_TO_ROAD_LANE";
                    compute_parking_control(world, sp->exit_merge_pt, sp->aisle_heading, GearState::DRIVE, 3.8f, false);
                    build_autopark_visual_ribbon(ego, (ego.pos + sp->exit_merge_pt) * 0.5f, sp->exit_merge_pt, sp->aisle_heading);
                }

                if ((ego.pos - sp->exit_merge_pt).len() < 1.85f ||
                    (ego.pos - sp->exit_merge_pt).dot(aisle_d) > -0.3f ||
                    maneuver_timer > 6.5f) {
                    ego.route_edges = {sp->road_edge_id};
                    ego.current_route_idx = 0;
                    ego.current_edge_s = sp->exit_road_s;
                    ego.current_lat_offset = 0.0f;
                    ego.gear = GearState::DRIVE;
                    world.ensure_route_horizon(ego);
                    autopark_state = AutoparkState::INACTIVE;
                    locked_parking_space_id = -1;
                }
            }
        }

        if (apply_controls_to_ego) {
            ego.steer_cmd = telemetry.expert_steer_cmd;
            ego.accel_cmd = telemetry.expert_accel_cmd;
        }
        return true;
    }

    void evaluate(CityWorld& world, float dt, bool apply_controls_to_ego) {
        Vehicle& ego = world.ego;
        telemetry = RuleTelemetry{};

        // 1. Update 360-degree Line-of-Sight perception mask
        world.update_360_perception_visibility();
        for (const auto& npc : world.npc_cars) if (npc.is_visible_to_ego) ++telemetry.visible_objects_360;
        for (const auto& pk  : world.parked_cars) if (pk.is_visible_to_ego) ++telemetry.visible_objects_360;
        for (const auto& bike : world.bicycles) if (bike.is_visible_to_ego) ++telemetry.visible_objects_360;
        for (const auto& ped : world.pedestrians) if (ped.is_visible_to_ego) ++telemetry.visible_objects_360;
        for (const auto& sp  : world.parking_spaces) {
            if (sp.is_visible_to_ego) {
                ++telemetry.visible_objects_360;
                if (!sp.is_occupied) ++telemetry.visible_free_parking_360;
            }
        }

        float cte = 0.0f, h_err = 0.0f;
        world.update_vehicle_route_progress(ego, &cte, &h_err);
        telemetry.cross_track_error = cte;
        telemetry.heading_error = h_err;

        world.get_route_lookahead(ego, 34.0f, planned_ribbon);
        if (planned_ribbon.empty()) return;

        // 2. If Autopark / Unpark is actively maneuvering or parked, run the Autopark controller
        if (step_autopark_state_machine(world, dt, apply_controls_to_ego)) {
            return;
        }

        Vec2 fwd = Vec2::from_angle(ego.heading);
        Vec2 left = fwd.perp_left();
        const LaneEdge& cur_edge = world.edges[ego.route_edges[ego.current_route_idx]];
        bool inside_inter = cur_edge.is_intersection_connector;
        float eff_cte_now = cte - ego.current_lat_offset;

        // 3. Proactive 3-Point K-Turn in Reverse (GearState::REVERSE):
        //    Only triggers when the car has genuinely overshot a tight turn well outside the lane
        //    (|eff_cte_now| > 2.05m and |h_err| > 0.85 rad pointing away from the lane) and cooldown has elapsed.
        //    Aligns nose with the 4.5m lookahead turn heading and enforces a 5.0s cooldown on completion so it NEVER oscillates
        if (k_turn_cooldown_timer > 0.0f) {
            k_turn_cooldown_timer = std::max(0.0f, k_turn_cooldown_timer - dt);
        }

        Vec2 check_pt = planned_ribbon.back().pos;
        float target_turn_h = planned_ribbon.back().heading;
        for (const auto& ps : planned_ribbon) {
            if (ps.s >= 4.2f) {
                check_pt = ps.pos + Vec2::from_angle(ps.heading).perp_left() * ego.current_lat_offset;
                target_turn_h = ps.heading;
                break;
            }
        }
        float turn_h_err = wrap_angle(ego.heading - target_turn_h);
        bool turn_overshoot = (k_turn_cooldown_timer <= 0.0f) &&
                              (std::fabs(eff_cte_now) > 2.05f) &&
                              (std::fabs(turn_h_err) > 0.85f) &&
                              ((turn_h_err * eff_cte_now) > 0.35f);

        if (k_turn_reverse_timer > 0.0f) {
            k_turn_reverse_timer = std::max(0.0f, k_turn_reverse_timer - dt);
            ego.gear = GearState::REVERSE;
            telemetry.reverse_maneuver_active = true;
            telemetry.primary_reason = "K_TURN_3_POINT_REVERSE";
            telemetry.lookahead_pt = check_pt;
            telemetry.lookahead_dist = 4.2f;

            if (has_close_obstacle_in_motion_dir(world, fwd * -1.0f, 3.6f)) {
                k_turn_reverse_timer = 0.0f;
                k_turn_cooldown_timer = 5.0f;
                ego.gear = GearState::DRIVE;
            } else {
                // In Reverse, steer opposite to turn_h_err (+turn_h_err) to pivot the nose toward the upcoming lane heading
                telemetry.expert_steer_cmd = clampf(turn_h_err * 1.60f - eff_cte_now * 0.25f, -1.0f, 1.0f);
                telemetry.target_speed = 1.85f;
                telemetry.expert_accel_cmd = (ego.speed < 1.85f) ? 0.58f : 0.0f;
                if ((k_turn_reverse_timer < 0.75f && std::fabs(turn_h_err) < 0.32f) || k_turn_reverse_timer <= 0.0f) {
                    k_turn_reverse_timer = 0.0f;
                    k_turn_cooldown_timer = 5.0f;
                    ego.gear = GearState::DRIVE;
                }
                if (apply_controls_to_ego) {
                    ego.steer_cmd = telemetry.expert_steer_cmd;
                    ego.accel_cmd = telemetry.expert_accel_cmd;
                }
                return;
            }
        } else if (turn_overshoot) {
            if (!has_close_obstacle_in_motion_dir(world, fwd * -1.0f, 4.0f)) {
                k_turn_reverse_timer = 1.60f;
                ego.gear = GearState::REVERSE;
                ego.speed *= 0.35f;
            }
        } else {
            ego.gear = GearState::DRIVE;
        }

        constexpr float CRUISE_SPEED = 11.2f;
        float v_desired = CRUISE_SPEED;
        float desired_lat_offset = 0.0f;

        // A. OVERTAKING & LANE-CHANGE ENGINE (Half-Parked Cars, Stopped Buses,
        //    Crashed Wrecks, and Slow Cyclists â€” with Hysteresis so it NEVER
        //    oscillates or cuts back into Lane 1 before fully clearing)
        bool obstacle_requires_overtake = false;
        float obstacle_dist_s = 99.0f;
        const char* overtake_reason_str = active_overtake_reason;

        if (inside_inter) {
            overtake_hold_timer = 0.0f;
        } else {
            // 1. Check Half-Parked Cars on our edge (keep active until rear bumper clears ds > -6.8m)
            for (const auto& pk : world.parked_cars) {
                if (pk.edge_id != cur_edge.id) continue;
                float ds = pk.s_pos - ego.current_edge_s;
                bool visible_or_passing = pk.is_visible_to_ego || (ds > -6.8f && ds < 6.0f && ego.current_lat_offset > 0.6f);
                if (!visible_or_passing) continue;
                if (ds > -6.8f && ds < 22.0f && ds < obstacle_dist_s) {
                    obstacle_requires_overtake = true;
                    obstacle_dist_s = ds;
                    telemetry.detected_parked_car_id = pk.id;
                    overtake_reason_str = "OVERTAKE_HALF_PARKED_CAR";
                }
            }

            // 2. Check Stopped City Transit Buses in "BUS" Zones or Crashed Wrecks on our edge
            //    (Use edge arc-length ds so passing alongside the 9.6m bus at ds in [-8.8m..0m] never drops detection)
            for (const auto& npc : world.npc_cars) {
                if (!npc.is_stopped_at_bus_zone && !npc.is_crashed) continue;
                int npc_edge_id = npc.route_edges[npc.current_route_idx];
                float clear_rear = npc.is_bus ? -9.2f : -6.8f;

                if (npc_edge_id == cur_edge.id) {
                    float ds = npc.current_edge_s - ego.current_edge_s;
                    bool visible_or_passing = npc.is_visible_to_ego || (ds > clear_rear && ds < 8.0f && ego.current_lat_offset > 0.6f);
                    if (visible_or_passing && ds > clear_rear && ds < 24.0f && ds < obstacle_dist_s) {
                        obstacle_requires_overtake = true;
                        obstacle_dist_s = ds;
                        if (npc.is_crashed) {
                            telemetry.crash_prevention_active = true;
                            telemetry.detected_crash_car_id = npc.id;
                            overtake_reason_str = "ACCIDENT_LOS_EVASIVE_BYPASS";
                        } else {
                            telemetry.detected_lead_car_id = npc.id;
                            overtake_reason_str = "OVERTAKE_STOPPED_BUS_ZONE";
                        }
                    }
                } else if (npc.is_crashed && npc.is_visible_to_ego) {
                    // Fallback for cross-lane / multi-car collision wreck blocking planned ribbon
                    for (const auto& ps : planned_ribbon) {
                        if (ps.s > 24.0f) break;
                        if ((npc.pos - ps.pos).len() < 2.55f) {
                            float ds = (npc.pos - ego.pos).dot(fwd);
                            if (ds > -6.5f && ds < 23.0f && ds < obstacle_dist_s) {
                                obstacle_requires_overtake = true;
                                obstacle_dist_s = ds;
                                telemetry.crash_prevention_active = true;
                                telemetry.detected_crash_car_id = npc.id;
                                overtake_reason_str = "ACCIDENT_LOS_EVASIVE_BYPASS";
                            }
                            break;
                        }
                    }
                }
            }

            // 3. Check slow Bicycles on open multi-lane or dotted overtaking stretches
            for (const auto& bike : world.bicycles) {
                int bike_edge_id = bike.route_edges[bike.current_route_idx];
                if (bike_edge_id != cur_edge.id) continue;
                float ds = bike.current_edge_s - ego.current_edge_s;
                bool visible_or_passing = bike.is_visible_to_ego || (ds > -5.5f && ds < 6.0f && ego.current_lat_offset > 0.6f);
                if (!visible_or_passing) continue;
                if (ds > -5.5f && ds < 16.0f && ds < obstacle_dist_s &&
                    (cur_edge.is_dotted_overtake_zone(ego.current_edge_s) || ego.current_lat_offset > 0.6f)) {
                    obstacle_requires_overtake = true;
                    obstacle_dist_s = ds;
                    telemetry.detected_bicycle_id = bike.id;
                    overtake_reason_str = "OVERTAKE_CYCLIST_LEFT";
                }
            }
        }

        // Only lock commitment when Ego is physically ALONGSIDE the obstacle (obstacle_dist_s < 4.8m and already in Lane 2).
        // Anytime Ego is still behind the obstacle (obstacle_dist_s >= 4.8m), it MUST abort and wait in Lane 1 if oncoming traffic appears
        bool already_alongside_obstacle = (ego.current_lat_offset > 1.25f && obstacle_dist_s < 4.8f && obstacle_dist_s > -7.2f);

        // Evaluate all overtaking lane checks in the ROAD CENTERLINE FRAME (road_fwd, road_left at ego.current_edge_s)
        // so Ego's instantaneous steering angle or yaw never corrupts lateral lane detection 30..55m ahead
        PathSample ego_road_ps = cur_edge.eval_at_s(ego.current_edge_s);
        Vec2 road_fwd  = Vec2::from_angle(ego_road_ps.heading);
        Vec2 road_left = road_fwd.perp_left();

        if (obstacle_requires_overtake) {
            float rem_road = cur_edge.length - ego.current_edge_s;
            bool legal_line_zone = (cur_edge.is_dotted_overtake_zone(ego.current_edge_s) && rem_road > 14.0f) ||
                                   already_alongside_obstacle ||
                                   (telemetry.crash_prevention_active && rem_road > 11.0f);

            // Dynamic time-to-complete-pass oncoming distance check (32m .. 56m depending on distance to obstacle)
            // On 4-lane boulevards (num_lanes_same_dir >= 2), Lane 2 (+2.65m) is a same-direction lane separated from
            // opposing traffic (road_lat >= 5.25m) by a double-yellow line, so only check road_lat < 3.85m
            bool oncoming_clear = true;
            float req_oncoming_dist = clampf((std::max(0.0f, obstacle_dist_s) + 12.0f) * 2.35f, 32.0f, 56.0f);
            float max_oncoming_lat = (cur_edge.num_lanes_same_dir >= 2) ? 3.85f : 6.6f;

            for (const auto& npc : world.npc_cars) {
                if (npc.is_crashed) continue;
                // Use dual-vantage (windshield + left B-pillar/mirror) LOS so a stopped bus in Lane 1 never hides oncoming cars in Lane 2
                if (!world.has_360_overtake_safety_los(npc.pos)) continue;

                Vec2 rel_road = npc.pos - ego_road_ps.pos;
                float road_lon = rel_road.dot(road_fwd);
                float road_lat = rel_road.dot(road_left);
                Vec2 nfwd = Vec2::from_angle(npc.heading);
                int npc_cur_edge = npc.route_edges[npc.current_route_idx];

                bool is_opposing_car = (nfwd.dot(road_fwd) < -0.30f) ||
                                       (cur_edge.oppose_edge_id >= 0 && npc_cur_edge == cur_edge.oppose_edge_id);
                if (is_opposing_car && road_lon > -4.0f && road_lon < req_oncoming_dist && road_lat > 0.4f && road_lat < max_oncoming_lat) {
                    // Standstill Tie-Breaker: If the opposing car has come to a complete stop (npc.speed < 0.45f) in its own lane
                    // (road_lat > 1.45f, not actively overtaking) well beyond our obstacle (road_lon > obstacle_dist_s + 5.5f)
                    // and Ego has already waited >= 1.5s, the opposing car is yielding to us â€” proceed with the pass
                    bool opposing_is_yielding_stationary = (npc.speed < 0.45f && !npc.is_overtaking &&
                                                            road_lat > 1.45f && road_lon > std::max(9.5f, obstacle_dist_s + 5.5f) &&
                                                            ego.intersection_wait_time > 1.5f);
                    if (!opposing_is_yielding_stationary) {
                        oncoming_clear = false;
                        telemetry.conflict_oncoming_car_id = npc.id;
                        break;
                    }
                }

                // Also check if an NPC inside the upcoming intersection is turning onto our opposing lane (on 2-lane roads)
                if (cur_edge.num_lanes_same_dir == 1 && cur_edge.oppose_edge_id >= 0 && road_lon > 0.0f && road_lon < req_oncoming_dist) {
                    for (int k = npc.current_route_idx; k < std::min(static_cast<int>(npc.route_edges.size()), npc.current_route_idx + 2); ++k) {
                        if (npc.route_edges[k] == cur_edge.oppose_edge_id) {
                            oncoming_clear = false;
                            telemetry.conflict_oncoming_car_id = npc.id;
                            break;
                        }
                    }
                    if (!oncoming_clear) break;
                }
            }

            // Check 360-degree Side-Repeater Blind Spot & Passing Lane Occupancy in the same direction (in Road Frame)
            bool blind_spot_clear = true;
            for (const auto& npc : world.npc_cars) {
                if (npc.is_crashed || npc.is_stopped_at_bus_zone) continue;
                if (!world.has_360_overtake_safety_los(npc.pos)) continue;
                Vec2 rel_road = npc.pos - ego_road_ps.pos;
                float road_lon = rel_road.dot(road_fwd);
                float road_lat = rel_road.dot(road_left);
                Vec2 nfwd = Vec2::from_angle(npc.heading);
                if (nfwd.dot(road_fwd) > 0.30f && road_lon > -15.0f && road_lon < 22.0f && road_lat > 0.85f && road_lat < 4.8f) {
                    blind_spot_clear = false;
                    break;
                }
            }

            if (legal_line_zone && (oncoming_clear || already_alongside_obstacle) && (blind_spot_clear || already_alongside_obstacle)) {
                desired_lat_offset = +2.65f;
                overtake_hold_timer = 0.65f; // Hold passing lane briefly after clearing obstacle before merging back
                active_overtake_reason = overtake_reason_str;
                telemetry.rule_overtaking_active = true;
                telemetry.primary_reason = overtake_reason_str;
                v_desired = std::min(v_desired, telemetry.crash_prevention_active ? 6.8f : 9.4f);

                // If already alongside the obstacle and an oncoming car approaches close (< 18m), brake to let it yield/clear safely
                if (already_alongside_obstacle && !oncoming_clear) {
                    v_desired = std::min(v_desired, 3.5f);
                }
            } else {
                // Unsafe to overtake. Return immediately to Lane 1 (0.0m) and brake safely behind the obstacle
                desired_lat_offset = 0.0f;
                overtake_hold_timer = 0.0f;
                telemetry.rule_waiting_to_overtake = true;
                telemetry.primary_reason = !oncoming_clear ? "WAIT_ONCOMING_FOR_OVERTAKE"
                                         : (!legal_line_zone ? "SOLID_LINE_NO_OVERTAKE" : "WAIT_BLIND_SPOT_OVERTAKE");
                float safe_gap = std::max(0.0f, obstacle_dist_s - ego.length - 2.4f);
                float v_wait = (safe_gap < 2.0f) ? 0.0f : std::sqrt(2.0f * 3.2f * safe_gap);
                v_desired = std::min(v_desired, v_wait);
            }
        } else if (overtake_hold_timer > 0.0f && !inside_inter) {
            // Post-overtake hold hysteresis: stay in Lane 2 for 0.65s after clearing before smoothly returning to Lane 1
            overtake_hold_timer = std::max(0.0f, overtake_hold_timer - dt);
            desired_lat_offset = +2.65f;
            telemetry.rule_overtaking_active = true;
            telemetry.primary_reason = active_overtake_reason;
        }

        // Smoothly interpolate Ego's lateral passing offset
        ego.current_lat_offset = lerpf(ego.current_lat_offset, desired_lat_offset, clampf(dt * 3.8f, 0.0f, 1.0f));
        ego.is_overtaking = (ego.current_lat_offset > 0.65f);
        telemetry.evasive_lat_offset = ego.current_lat_offset;

        // B. 360-DEGREE DYNAMIC COLLISION PREVENTION (Surround TTC < 1.65s)
        Vec2 ego_vel = fwd * ego.speed;
        for (const auto& npc : world.npc_cars) {
            if (!npc.is_visible_to_ego || npc.is_crashed) continue;

            Vec2 rel_p = npc.pos - ego.pos;
            float dist = rel_p.len();
            if (dist > 18.0f || dist < 0.1f) continue;

            Vec2 npc_vel = Vec2::from_angle(npc.heading) * npc.speed;
            Vec2 rel_v = npc_vel - ego_vel;
            float v_close = -(rel_p.dot(rel_v)) / dist;

            if (v_close > 3.2f) {
                float v_sq = std::max(0.1f, rel_v.len_sq());
                float t_cpa = clampf(-(rel_p.dot(rel_v)) / v_sq, 0.0f, 3.0f);
                Vec2 closest_rel = rel_p + rel_v * t_cpa;
                float d_cpa = closest_rel.len();

                if (d_cpa < 2.35f && t_cpa < 1.60f) {
                    telemetry.crash_prevention_active = true;
                    telemetry.conflict_oncoming_car_id = npc.id;

                    float lon_pos = rel_p.dot(fwd);
                    if (lon_pos < -2.0f) {
                        telemetry.primary_reason = "360_REAR_CRASH_PREVENTION";
                        v_desired = std::max(v_desired, std::min(CRUISE_SPEED + 2.0f, ego.speed + 3.0f));
                    } else {
                        telemetry.primary_reason = "360_CRASH_PREVENTION_AEB";
                        v_desired = 0.0f;
                    }
                    break;
                }
            }
        }

        // C. Pure pursuit steering + lateral overtaking offset + Stanley recovery
        //    (Use a tighter lookahead chord inside roundabouts/intersections so the
        //     vehicle tracks tight right turns & circular lanes accurately)
        float ld = inside_inter ? clampf(1.85f + 0.16f * ego.speed, 1.85f, 2.95f)
                                : clampf(3.4f + 0.45f * ego.speed, 3.4f, 9.2f);
        telemetry.lookahead_dist = ld;

        Vec2 target_pt = planned_ribbon.back().pos;
        float target_heading = planned_ribbon.back().heading;
        for (const auto& ps : planned_ribbon) {
            if (ps.s >= ld) {
                target_pt = ps.pos;
                target_heading = ps.heading;
                break;
            }
        }
        if (std::fabs(ego.current_lat_offset) > 0.05f) {
            Vec2 target_left = Vec2::from_angle(target_heading).perp_left();
            target_pt += target_left * ego.current_lat_offset;
        }
        telemetry.lookahead_pt = target_pt;

        Vec2 to_target = target_pt - ego.pos;
        float local_y = to_target.dot(left);
        float dist_sq = std::max(0.95f, to_target.len_sq());

        float pp_steer_rad = std::atan(2.0f * ego.wheelbase * local_y / dist_sq);
        float eff_cte = cte - ego.current_lat_offset;
        float recovery_rad = -0.28f * eff_cte - 0.42f * h_err;
        float total_steer_rad = pp_steer_rad + recovery_rad;
        float max_steer_norm = (ego.speed < 5.8f) ? 0.68f : 0.58f;
        telemetry.expert_steer_cmd = clampf(total_steer_rad / max_steer_norm, -1.0f, 1.0f);

        // D. Corner grip speed profiler
        constexpr float SAFE_LAT_ACCEL = 4.5f;
        constexpr float COMF_DECEL = 3.4f;
        float min_corner_v = CRUISE_SPEED;

        for (const auto& ps : planned_ribbon) {
            float abs_k = std::fabs(ps.curvature);
            if (abs_k > 0.035f && ps.s < 24.0f) {
                float v_curve_max = std::sqrt(SAFE_LAT_ACCEL / abs_k);
                float dist_ahead = std::max(0.0f, ps.s - 1.5f);
                float v_allow = std::sqrt(v_curve_max * v_curve_max + 2.0f * COMF_DECEL * dist_ahead);
                if (v_allow < min_corner_v) {
                    min_corner_v = v_allow;
                }
            }
        }
        telemetry.corner_speed_limit = min_corner_v;
        if (min_corner_v < v_desired - 0.4f) {
            v_desired = min_corner_v;
            telemetry.rule_corner_grip_limit = true;
            if (!telemetry.crash_prevention_active && !telemetry.rule_overtaking_active && !telemetry.rule_waiting_to_overtake) {
                telemetry.primary_reason = "CORNER_GRIP_PHYSICS";
            }
        }

        // E. Intersection control: lights, police officer, stop & yield signs
        int upcoming_inter_id = cur_edge.to_node;
        const Intersection& upcoming_inter = world.intersections[upcoming_inter_id];
        int approach_dir = cur_edge.exit_dir;
        float d_stop = 99.0f;

        if (!inside_inter) {
            d_stop = std::max(0.0f, cur_edge.length - ego.current_edge_s);
        }
        telemetry.dist_to_stop_line = d_stop;

        bool inter_visible = world.has_360_line_of_sight(upcoming_inter.center);
        if (!inside_inter && d_stop < 28.0f && inter_visible) {
            if (upcoming_inter.uses_phase_signal()) {
                float t_rem = 0.0f;
                LightState ls = upcoming_inter.get_light_for_approach(approach_dir, &t_rem);
                bool is_police = upcoming_inter.is_police_controlled();

                constexpr float T_REACT = 0.22f;
                constexpr float A_YELLOW_BRAKE = 4.4f;
                float d_min_stop = ego.speed * T_REACT + (ego.speed * ego.speed) / (2.0f * A_YELLOW_BRAKE);
                float d_max_clear = std::max(ego.speed, 8.0f) * t_rem - 2.5f;

                telemetry.dilemma_min_stop_dist = d_min_stop;
                telemetry.dilemma_max_clear_dist = d_max_clear;

                if (ls == LightState::RED) {
                    telemetry.rule_red_light_stop = !is_police;
                    telemetry.rule_police_gesture = is_police;
                    if (!telemetry.crash_prevention_active) {
                        telemetry.primary_reason = is_police ? "POLICE_OFFICER_STOP_PALM" : "RED_LIGHT_STOP";
                    }
                    float stop_margin = std::max(0.0f, d_stop - 1.6f);
                    float v_red = (stop_margin < 0.9f) ? 0.0f : std::sqrt(2.0f * 2.8f * stop_margin);
                    v_desired = std::min(v_desired, v_red);
                } else if (ls == LightState::YELLOW) {
                    telemetry.rule_police_gesture = is_police;
                    if (d_stop < d_min_stop && d_stop <= d_max_clear && d_stop < 11.0f && ego.speed > 4.5f) {
                        telemetry.rule_dilemma_commit = true;
                        if (!telemetry.crash_prevention_active) {
                            telemetry.primary_reason = is_police ? "POLICE_CLEARANCE_PROCEED" : "YELLOW_DILEMMA_COMMIT";
                        }
                        v_desired = std::max(v_desired, std::min(CRUISE_SPEED, ego.speed + 0.8f));
                    } else {
                        telemetry.rule_dilemma_stop = true;
                        if (!telemetry.crash_prevention_active) {
                            telemetry.primary_reason = is_police ? "POLICE_WHISTLE_HOLD" : "YELLOW_DILEMMA_STOP";
                        }
                        float stop_margin = std::max(0.0f, d_stop - 1.5f);
                        float v_yel = (stop_margin < 0.9f) ? 0.0f : std::sqrt(2.0f * 3.4f * stop_margin);
                        v_desired = std::min(v_desired, v_yel);
                    }
                } else if (is_police && d_stop < 14.0f && !telemetry.crash_prevention_active && !telemetry.rule_overtaking_active) {
                    telemetry.rule_police_gesture = true;
                    telemetry.primary_reason = "POLICE_OFFICER_WAVE_GO";
                }
            } else if (upcoming_inter.has_stop_sign_for_approach(approach_dir)) {
                if (!ego.stop_sign_completed_for_edge && d_stop < 20.0f) {
                    telemetry.rule_stop_sign = true;
                    if (!telemetry.crash_prevention_active) {
                        telemetry.primary_reason = "FOUR_WAY_STOP_SIGN";
                    }
                    float stop_margin = std::max(0.0f, d_stop - 1.8f);
                    if (stop_margin < 1.1f) {
                        v_desired = 0.0f;
                        if (ego.speed < 0.45f) {
                            ego.stop_sign_hold_timer += dt;
                            if (ego.stop_sign_hold_timer >= 0.55f) {
                                ego.stop_sign_completed_for_edge = true;
                            }
                        }
                    } else {
                        v_desired = std::min(v_desired, std::sqrt(2.0f * 2.8f * stop_margin));
                    }
                }
            } else if (upcoming_inter.has_yield_sign_for_approach(approach_dir)) {
                if (d_stop < 18.0f) {
                    bool major_conflict = false;
                    for (const auto& npc : world.npc_cars) {
                        if (!npc.is_visible_to_ego) continue;
                        float d_inter = (npc.pos - upcoming_inter.center).len();
                        if (d_inter < 19.0f && npc.speed > 1.0f) {
                            Vec2 nfwd = Vec2::from_angle(npc.heading);
                            if (std::fabs(nfwd.dot(fwd)) < 0.45f && (upcoming_inter.center - npc.pos).dot(nfwd) > -2.5f) {
                                major_conflict = true;
                                telemetry.conflict_oncoming_car_id = npc.id;
                                break;
                            }
                        }
                    }
                    if (major_conflict) {
                        telemetry.rule_yield_sign = true;
                        if (!telemetry.crash_prevention_active) {
                            telemetry.primary_reason = "YIELD_SIGN_CROSS_TRAFFIC";
                        }
                        float stop_margin = std::max(0.0f, d_stop - 1.8f);
                        float v_yld = (stop_margin < 1.0f) ? 0.0f : std::sqrt(2.0f * 2.8f * stop_margin);
                        v_desired = std::min(v_desired, v_yld);
                    }
                }
            }
        }

        // F. Unprotected left turn oncoming traffic yield (360 LOS filtered)
        bool turning_left_soon = (ego.active_intent == TurnIntent::LEFT) && (d_stop < 8.5f || inside_inter);
        if (turning_left_soon) {
            int opp_dir = (approach_dir + 2) & 3;
            Vec2 inter_center = upcoming_inter.center;

            for (const auto& npc : world.npc_cars) {
                if (!npc.is_visible_to_ego || npc.is_crashed) continue;
                float dist_to_inter = (npc.pos - inter_center).len();
                if (dist_to_inter > 24.0f) continue;

                Vec2 npc_fwd = Vec2::from_angle(npc.heading);
                Vec2 opp_vec = dir_vec(opp_dir);
                if (npc_fwd.dot(opp_vec) > 0.65f && npc.active_intent != TurnIntent::LEFT) {
                    Vec2 to_inter = inter_center - npc.pos;
                    if (to_inter.dot(npc_fwd) > -4.0f) {
                        float npc_tti = dist_to_inter / std::max(npc.speed, 2.5f);
                        if (npc_tti < 3.4f || dist_to_inter < 11.5f) {
                            if (ego.intersection_wait_time > 3.5f && npc.speed < 0.8f) {
                                telemetry.rule_deadlock_breaker = true;
                                telemetry.primary_reason = "DEADLOCK_TIE_BREAKER";
                            } else {
                                telemetry.rule_unprotected_left_yield = true;
                                telemetry.conflict_oncoming_car_id = npc.id;
                                telemetry.primary_reason = "UNPROTECTED_LEFT_YIELD";
                                float creep_v = (d_stop > 2.2f && !inside_inter) ? 2.2f : 0.0f;
                                v_desired = std::min(v_desired, creep_v);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // G. Predictive pedestrian & red-light runner AEB (360 LOS)
        for (const auto& ped : world.pedestrians) {
            if (!ped.active_crossing || !ped.is_visible_to_ego) continue;
            float dist_to_ego = (ped.pos - ego.pos).len();
            if (dist_to_ego > 24.0f) continue;

            Vec2 ped_mid = ped.pos + ped.vel * 0.55f;
            Vec2 ped_fut = ped.pos + ped.vel * 1.15f;
            float corridor_r = ped.is_red_light_runner ? 3.1f : 2.65f;

            for (const auto& ps : planned_ribbon) {
                if (ps.s > 20.0f) break;
                float d0 = (ped.pos - ps.pos).len();
                float d1 = (ped_mid - ps.pos).len();
                float d2 = (ped_fut - ps.pos).len();
                if (std::min({d0, d1, d2}) < corridor_r && ps.s > 0.4f) {
                    telemetry.rule_pedestrian_aeb = true;
                    telemetry.detected_pedestrian_id = ped.id;
                    if (ped.is_red_light_runner) telemetry.crash_prevention_active = true;
                    telemetry.primary_reason = ped.is_red_light_runner ? "RED_RUNNER_EMERGENCY_AEB" : "PEDESTRIAN_CROSSWALK_YIELD";
                    float safe_d = std::max(0.0f, ps.s - (ped.is_red_light_runner ? 4.8f : 3.8f));
                    float v_ped = (safe_d < 1.5f) ? 0.0f : std::sqrt(2.0f * 4.2f * safe_d);
                    v_desired = std::min(v_desired, v_ped);
                    break;
                }
            }
        }

        // H. Bicycle corridor protection (when not overtaking on the left)
        if (!ego.is_overtaking) {
            for (const auto& bike : world.bicycles) {
                if (!bike.is_visible_to_ego) continue;
                Vec2 rel = bike.pos - ego.pos;
                if (rel.len() > 20.0f || rel.dot(fwd) < 0.5f) continue;

                for (const auto& ps : planned_ribbon) {
                    if (ps.s < 1.0f || ps.s > 17.0f) continue;
                    if ((bike.pos - ps.pos).len() < 1.75f) {
                        telemetry.rule_bicycle_follow = true;
                        telemetry.detected_bicycle_id = bike.id;
                        float gap = std::max(0.0f, ps.s - ego.length - 1.5f);
                        float v_bike = (gap < 1.5f) ? 0.0f : std::min(CRUISE_SPEED, bike.speed + 0.35f * gap);
                        if (v_bike < v_desired - 0.05f) {
                            v_desired = v_bike;
                            telemetry.primary_reason = "BICYCLE_SAFE_FOLLOW";
                        }
                        break;
                    }
                }
            }
        }

        // I. Intelligent Driver Model (IDM) lead vehicle ACC (360 LOS filtered)
        float closest_lead_s = 99.0f;
        Vehicle* lead_car = nullptr;

        for (auto& npc : world.npc_cars) {
            if (!npc.is_visible_to_ego || npc.is_crashed) continue;
            // If we are actively overtaking a stopped bus in the left lane, don't brake for the bus in the right lane
            if (ego.is_overtaking && npc.is_stopped_at_bus_zone) continue;

            Vec2 rel_ego = npc.pos - ego.pos;
            float direct_d = rel_ego.len();
            if (direct_d > 22.0f || rel_ego.dot(fwd) < 0.5f) continue;
            // Lateral alignment with Ego's current lane offset
            if (std::fabs(rel_ego.dot(left)) > 1.75f && !inside_inter) continue;

            Vec2 npc_fwd = Vec2::from_angle(npc.heading);

            for (const auto& ps : planned_ribbon) {
                if (ps.s < 1.2f || ps.s > 20.0f) continue;
                Vec2 rib_dir = Vec2::from_angle(ps.heading);
                if (npc_fwd.dot(rib_dir) < -0.15f) continue;
                Vec2 shifted_ps = ps.pos + rib_dir.perp_left() * ego.current_lat_offset;
                if ((npc.pos - shifted_ps).len() < 1.85f) {
                    if (ps.s < closest_lead_s) {
                        closest_lead_s = ps.s;
                        lead_car = &npc;
                    }
                    break;
                }
            }
        }

        if (lead_car != nullptr) {
            Vec2 npc_fwd = Vec2::from_angle(lead_car->heading);
            bool same_direction = (npc_fwd.dot(fwd) > 0.35f);

            if (!same_direction && ego.intersection_wait_time > 2.8f && lead_car->speed < 0.4f && closest_lead_s > 4.5f) {
                telemetry.rule_deadlock_breaker = true;
                telemetry.primary_reason = "DEADLOCK_TIE_BREAKER";
            } else {
                telemetry.rule_lead_vehicle_idm = true;
                telemetry.detected_lead_car_id = lead_car->id;

                float lead_half_len = lead_car->length * 0.5f;
                float ego_half_len  = ego.length * 0.5f;
                float net_gap = std::max(0.1f, closest_lead_s - ego_half_len - lead_half_len - 1.2f);
                float dv = ego.speed - lead_car->speed;
                constexpr float S0 = 2.2f;
                constexpr float TIME_HEADWAY = 1.15f;
                float s_star = S0 + std::max(0.0f, ego.speed * TIME_HEADWAY + (ego.speed * dv) / (2.0f * std::sqrt(3.5f * 4.5f)));
                float v_idm = (net_gap < 1.4f) ? 0.0f : CRUISE_SPEED * clampf(net_gap / std::max(s_star, 2.5f), 0.0f, 1.0f);
                if (v_idm < v_desired) {
                    v_desired = v_idm;
                    if (!telemetry.crash_prevention_active && !telemetry.rule_overtaking_active && !telemetry.rule_waiting_to_overtake) {
                        telemetry.primary_reason = lead_car->is_bus ? "FOLLOW_TRANSIT_BUS_IDM" : "LEAD_VEHICLE_IDM_ACC";
                    }
                }

                LightState approach_ls = inside_inter
                    ? LightState::GREEN
                    : upcoming_inter.get_light_for_approach(approach_dir);
                if (!lead_car->is_stopped_at_bus_zone && ego.intersection_wait_time > 3.5f &&
                    lead_car->speed < 0.35f && approach_ls == LightState::GREEN) {
                    lead_car->yield_impatience = 5.0f;
                }
            }
        }

        if (ego.speed < 0.45f) {
            ego.intersection_wait_time += dt;
        } else if (ego.speed > 2.0f) {
            ego.intersection_wait_time = 0.0f;
        }

        // J. Longitudinal acceleration / braking controller
        if (autopark_state == AutoparkState::SEARCHING_360_LOS) {
            v_desired = std::min(v_desired, 8.2f);
            if (!telemetry.crash_prevention_active && !telemetry.rule_red_light_stop && !telemetry.rule_pedestrian_aeb) {
                telemetry.primary_reason = "AUTOPARK_SCANNING_360_LOS";
            }
        } else if (autopark_state == AutoparkState::APPROACHING_AISLE) {
            v_desired = std::min(v_desired, 4.8f);
            if (!telemetry.crash_prevention_active && !telemetry.rule_red_light_stop && !telemetry.rule_pedestrian_aeb) {
                telemetry.primary_reason = "AUTOPARK_APPROACH_ON_ROAD";
            }
        }
        telemetry.target_speed = v_desired;
        float speed_err = v_desired - ego.speed;
        float accel_out = 0.0f;
        if (v_desired < 0.15f && ego.speed < 0.8f) {
            accel_out = -0.92f;
        } else if (speed_err >= 0.0f) {
            accel_out = clampf(speed_err * 0.40f, 0.0f, 0.85f);
        } else {
            accel_out = clampf(speed_err * (telemetry.crash_prevention_active ? 0.72f : 0.52f), -1.0f, 0.0f);
        }
        telemetry.expert_accel_cmd = accel_out;

        if (apply_controls_to_ego) {
            ego.steer_cmd = telemetry.expert_steer_cmd;
            ego.accel_cmd = telemetry.expert_accel_cmd;
        }
    }
};

} // namespace sim
