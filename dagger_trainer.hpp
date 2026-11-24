#pragma once
// Online imitation learning controller (DAgger).
// Pairs the rule-based driving planner with the convolutional neural policy.
#include "rule_driver_v11.hpp"
#include "cnn_engine_v12.hpp"

namespace sim {

enum class DriveMode : int {
    RULE_V11_EXPERT = 0,       // Rule-based planner (pure pursuit, IDM, traffic rules)
    DAGGER_TRAINING = 1,       // Online learning (CNN drives while the rule planner labels & supervises)
    NEURAL_V12_AUTONOMOUS = 2, // End-to-end CNN control
    MANUAL_HUMAN = 3           // Manual keyboard control (WASD / Arrow keys)
};

struct DAggerSample {
    std::array<float, TOTAL_IMG_FLOATS> img;
    std::array<float, AUX_DIM> aux;
    float expert_steer = 0.0f;
    float expert_accel = 0.0f;
    float weight = 1.0f;
};

class AutonomyController {
public:
    DriveMode mode = DriveMode::DAGGER_TRAINING;
    RuleBasedDriverV11 expert_v11;
    EndToEndCNNv12 cnn_v12;

    static constexpr size_t MAX_BUFFER_SIZE = 1200;
    std::vector<DAggerSample> replay_buffer;
    size_t buffer_write_head = 0;

    std::vector<float> loss_history;
    std::vector<float> disagreement_history;
    float smoothed_loss = 0.35f;
    float current_disagreement = 0.0f;
    int total_interventions = 0;
    int dagger_iterations = 0;
    float autonomous_meters_since_intervention = 0.0f;
    float best_autonomous_run_meters = 0.0f;
    float intervention_flash_timer = 0.0f;
    const char* last_intervention_reason = "NONE";

    // Short horizon rollout for visualizing the CNN's current steering command
    std::vector<Vec2> cnn_predicted_ribbon;

    FastRNG train_rng{0xDA66E201u};
    int frame_counter = 0;

    AutonomyController() {
        replay_buffer.reserve(MAX_BUFFER_SIZE);
        loss_history.reserve(300);
        disagreement_history.reserve(300);
    }

    void reset_cnn_to_untrained() {
        FastRNG init_rng( static_cast<uint32_t>(frame_counter + 0x99112233u) );
        cnn_v12.init_weights(init_rng);
        replay_buffer.clear();
        buffer_write_head = 0;
        loss_history.clear();
        disagreement_history.clear();
        smoothed_loss = 0.45f;
        total_interventions = 0;
        dagger_iterations = 0;
        autonomous_meters_since_intervention = 0.0f;
    }

    void record_dagger_sample(const CityWorld& world, const std::array<float, AUX_DIM>& aux,
                              float exp_steer, float exp_accel, float priority_weight) {
        DAggerSample s;
        s.img = world.camera_tensor;
        s.aux = aux;
        s.expert_steer = exp_steer;
        s.expert_accel = exp_accel;
        s.weight = clampf(priority_weight, 0.8f, 3.0f);

        if (replay_buffer.size() < MAX_BUFFER_SIZE) {
            replay_buffer.push_back(s);
        } else {
            replay_buffer[buffer_write_head] = s;
            buffer_write_head = (buffer_write_head + 1) % MAX_BUFFER_SIZE;
        }
    }

    void train_minibatch(int batch_size = 10, bool force_latest = true) {
        if (replay_buffer.size() < 8) return;

        cnn_v12.zero_all_grads();
        float batch_loss = 0.0f;
        int n = static_cast<int>(replay_buffer.size());

        // Include the latest frame (when newly recorded) + random replay samples
        for (int b = 0; b < batch_size; ++b) {
            int idx = (b == 0 && force_latest)
                        ? ((buffer_write_head == 0) ? (n - 1) : static_cast<int>(buffer_write_head - 1))
                        : train_rng.range_int(0, n - 1);
            if (idx < 0 || idx >= n) idx = n - 1;
            const auto& sample = replay_buffer[idx];
            batch_loss += cnn_v12.accumulate_sample_grad(
                sample.img.data(), sample.aux.data(),
                sample.expert_steer, sample.expert_accel, sample.weight
            );
        }

        cnn_v12.apply_adamw_step(batch_size);
        ++dagger_iterations;

        batch_loss /= static_cast<float>(batch_size);
        cnn_v12.last_loss = batch_loss;
        smoothed_loss = (dagger_iterations == 1) ? batch_loss : lerpf(smoothed_loss, batch_loss, 0.08f);

        if (loss_history.size() >= 240) {
            loss_history.erase(loss_history.begin());
        }
        loss_history.push_back(smoothed_loss);
    }

    // Build predicted kinematic trajectory arc for the CNN's current steering command
    void update_cnn_predicted_ribbon(const Vehicle& ego) {
        cnn_predicted_ribbon.clear();
        Vec2 p = ego.pos;
        float h = ego.heading;
        float steer_rad = cnn_v12.predicted_steer * 0.58f;
        float v = std::max(4.5f, ego.speed + cnn_v12.predicted_accel * 2.0f);
        float dt_step = 0.14f;

        cnn_predicted_ribbon.push_back(p);
        for (int i = 0; i < 18; ++i) {
            float yaw_rate = (v / ego.wheelbase) * std::tan(steer_rad);
            p += Vec2::from_angle(h) * (v * dt_step);
            h = wrap_angle(h + yaw_rate * dt_step);
            cnn_predicted_ribbon.push_back(p);
        }
    }

    // Advance world entities, sensors, rule planner, and neural policy by dt
    void step(CityWorld& world, float dt, float manual_steer = 0.0f, float manual_accel = 0.0f, bool compute_visual_saliency = true) {
        ++frame_counter;
        if (intervention_flash_timer > 0.0f) {
            intervention_flash_timer = std::max(0.0f, intervention_flash_timer - dt);
        }

        for (auto& inter : world.intersections) {
            inter.step(dt);
        }
        world.step_pedestrians(dt);
        world.step_bicycles(dt);
        world.step_npc_vehicles(dt);

        // Render ego vehicle's dual onboard camera feeds
        world.render_dual_cameras();

        // Run the rule-based planner in shadow mode to get reference controls and telemetry
        expert_v11.evaluate(world, dt, false);
        const RuleTelemetry& exp = expert_v11.telemetry;

        // Forward pass through the CNN
        std::array<float, AUX_DIM> aux = EndToEndCNNv12::build_aux_vector(world, exp);
        cnn_v12.forward(world.camera_tensor.data(), aux.data());
        if (compute_visual_saliency) {
            cnn_v12.compute_saliency_maps();
        }
        update_cnn_predicted_ribbon(world.ego);

        float steer_diff = std::fabs(cnn_v12.predicted_steer - exp.expert_steer_cmd);
        float accel_diff = std::fabs(cnn_v12.predicted_accel - exp.expert_accel_cmd);
        current_disagreement = 0.65f * steer_diff + 0.35f * accel_diff;
        if (disagreement_history.size() >= 240) {
            disagreement_history.erase(disagreement_history.begin());
        }
        disagreement_history.push_back(current_disagreement);

        bool active_parking_or_reverse = (expert_v11.autopark_state != AutoparkState::INACTIVE &&
                                          expert_v11.autopark_state != AutoparkState::SEARCHING_360_LOS) ||
                                         exp.reverse_maneuver_active;
        bool in_inter_turn = world.edges[world.ego.route_edges[world.ego.current_route_idx]].is_intersection_connector;

        if (active_parking_or_reverse || mode == DriveMode::RULE_V11_EXPERT) {
            world.ego.steer_cmd = exp.expert_steer_cmd;
            world.ego.accel_cmd = exp.expert_accel_cmd;
            if (!active_parking_or_reverse && frame_counter % 3 == 0 && replay_buffer.size() < 250) {
                record_dagger_sample(world, aux, exp.expert_steer_cmd, exp.expert_accel_cmd, 1.0f);
            }
        } else if (mode == DriveMode::DAGGER_TRAINING) {
            // Let the CNN drive to explore its own state distribution, but blend in expert
            // guidance during overtakes or tight intersection turns if it drifts off the arc.
            float effective_cte = exp.cross_track_error - exp.evasive_lat_offset;
            bool overtake_safety_critical = exp.rule_waiting_to_overtake || exp.rule_overtaking_active || exp.crash_prevention_active;
            bool turn_guidance_needed = in_inter_turn && (std::fabs(effective_cte) > 0.75f || steer_diff > 0.45f);
            float beta = overtake_safety_critical ? 0.85f
                       : (turn_guidance_needed    ? 0.80f
                       : ((replay_buffer.size() < 60) ? 0.45f : 0.0f));
            world.ego.steer_cmd = lerpf(cnn_v12.predicted_steer, exp.expert_steer_cmd, beta);
            world.ego.accel_cmd = lerpf(cnn_v12.predicted_accel, exp.expert_accel_cmd, beta);

            float weight = 1.0f + 1.5f * steer_diff + 0.8f * accel_diff;
            if (exp.rule_pedestrian_aeb || exp.rule_red_light_stop || exp.rule_dilemma_stop || exp.rule_unprotected_left_yield) {
                weight += 0.8f;
            }

            // Throttle sample recording when stationary at a red light so stopped frames don't dominate the buffer
            bool is_stopped_hold = (world.ego.speed < 0.35f && exp.expert_accel_cmd < 0.0f);
            bool should_record = is_stopped_hold ? (frame_counter % 25 == 0)
                                                 : (frame_counter % 2 == 0 || current_disagreement > 0.14f);
            if (should_record) {
                record_dagger_sample(world, aux, exp.expert_steer_cmd, exp.expert_accel_cmd, weight);
            }

            // Supervisor takeover check (measured relative to the active target lane offset)
            bool off_lane = (!in_inter_turn && std::fabs(effective_cte) > 1.95f) ||
                            ( in_inter_turn && std::fabs(effective_cte) > 2.25f) ||
                            (!in_inter_turn && std::fabs(exp.heading_error) > 0.90f);
            bool missed_aeb = ((exp.rule_pedestrian_aeb || exp.rule_red_light_stop) &&
                               exp.expert_accel_cmd < -0.55f && cnn_v12.predicted_accel > 0.05f && world.ego.speed > 3.5f);

            if (off_lane || missed_aeb) {
                ++total_interventions;
                intervention_flash_timer = 0.85f;
                last_intervention_reason = off_lane ? "LANE_DEPARTURE_RECOVERY" : "SAFETY_AEB_OVERRIDE";
                autonomous_meters_since_intervention = 0.0f;

                record_dagger_sample(world, aux, exp.expert_steer_cmd, exp.expert_accel_cmd, 2.8f);

                world.ego.steer_cmd = exp.expert_steer_cmd;
                world.ego.accel_cmd = exp.expert_accel_cmd;
            }

            if (!is_stopped_hold || frame_counter % 6 == 0) {
                train_minibatch(8, should_record);
                cnn_v12.forward(world.camera_tensor.data(), aux.data());
            }
        } else if (mode == DriveMode::NEURAL_V12_AUTONOMOUS) {
            // End-to-end neural control with physical steering fallback if severely off-track
            float effective_cte = exp.cross_track_error - exp.evasive_lat_offset;
            if (std::fabs(effective_cte) > 2.4f || std::fabs(exp.heading_error) > 0.85f) {
                world.ego.steer_cmd = exp.expert_steer_cmd;
                world.ego.accel_cmd = exp.expert_accel_cmd;
            } else {
                world.ego.steer_cmd = cnn_v12.predicted_steer;
                world.ego.accel_cmd = cnn_v12.predicted_accel;
            }
        } else if (mode == DriveMode::MANUAL_HUMAN) {
            world.ego.steer_cmd = manual_steer;
            world.ego.accel_cmd = manual_accel;
        }

        world.ego.step_physics(dt);
        world.sim_time += dt;

        if (mode == DriveMode::DAGGER_TRAINING || mode == DriveMode::NEURAL_V12_AUTONOMOUS) {
            autonomous_meters_since_intervention += world.ego.speed * dt;
            if (autonomous_meters_since_intervention > best_autonomous_run_meters) {
                best_autonomous_run_meters = autonomous_meters_since_intervention;
            }
        }
    }

    // Run a batch of simulation + training steps back-to-back
    void run_turbo_dagger_training(CityWorld& world, int num_steps = 220) {
        DriveMode prev_mode = mode;
        mode = DriveMode::DAGGER_TRAINING;
        for (int i = 0; i < num_steps; ++i) {
            step(world, 0.05f, 0.0f, 0.0f, (i == num_steps - 1));
        }
        mode = prev_mode;
    }
};

} // namespace sim
