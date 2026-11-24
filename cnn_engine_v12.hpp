#pragma once
// End-to-end convolutional policy network built on top of EngineZero.
// Takes dual-camera RGB frames + 8 kinematics/navigation cues and outputs [steer, accel].
#include "city_world.hpp"
#include "rule_driver_v11.hpp"
#include "engine_zero/Tensor.hpp"
#include "engine_zero/Autograd.hpp"
#include "engine_zero/Ops.hpp"
#include "engine_zero/Module.hpp"
#include "engine_zero/Conv2d.hpp"
#include "engine_zero/Linear.hpp"
#include "engine_zero/Optimizer.hpp"
#include <cstdio>

namespace sim {

constexpr int AUX_DIM = 8;

// Cached layer activations for the UI inspector panel
struct HudConvView {
    int w_out = CAM_W / 2;
    int h_out = CAM_H / 2;
    std::vector<float> pool_out;
};

struct HudDenseView {
    std::vector<float> out_act;
};

class EndToEndCNNv12 : public Module {
public:
    // EngineZero Modules
    Conv2D conv1_layer; // 6 x 24 x 32 -> 10 x 12 x 16
    Conv2D conv2_layer; // 10 x 12 x 16 -> 16 x 6 x 8
    Conv2D conv3_layer; // 16 x 6 x 8 -> 20 x 3 x 4 (= 240)

    Linear fc1_layer;   // (240 + 8) = 248 -> 64
    Linear fc2_layer;   // (64 + 8) = 72 -> 32 (HydraNet skip fusion)
    Linear fc_out_layer;// 32 -> 2 ([steer_cmd, accel_cmd])

    AdamW optimizer;

    // Last autograd graph endpoints for live Saliency Map backward pass
    ValuePtr last_img_node;
    ValuePtr last_out_node;

    // HUD feature map inspection caches
    HudConvView conv1;
    HudDenseView fc1;

    // Real-time spatial Saliency Maps [CAM_H * CAM_W] for Camera 0 & Camera 1, normalized in [0, 1]
    std::array<float, CAM_W * CAM_H> saliency_cam0{};
    std::array<float, CAM_W * CAM_H> saliency_cam1{};

    uint64_t adam_step_count = 0;
    float learning_rate = 0.0028f;
    float last_loss = 0.0f;
    float predicted_steer = 0.0f;
    float predicted_accel = 0.0f;

    EndToEndCNNv12() {
        FastRNG rng(0x2026F5D1u);
        init_weights(rng);
    }

    void init_weights(FastRNG& rng) {
        conv1_layer.init(TOTAL_IMG_CHANNELS, 10, 3, 1, 1, rng.next_u32());
        conv2_layer.init(10, 16, 3, 1, 1, rng.next_u32());
        conv3_layer.init(16, 20, 3, 1, 1, rng.next_u32());

        size_t flat_vis = 20 * (CAM_H / 8) * (CAM_W / 8); // 240
        fc1_layer.init(flat_vis + AUX_DIM, 64, rng.next_u32());
        fc2_layer.init(64 + AUX_DIM, 32, rng.next_u32());
        fc_out_layer.init(32, 2, rng.next_u32());

        conv1.w_out = CAM_W / 2;
        conv1.h_out = CAM_H / 2;
        conv1.pool_out.assign(10 * (CAM_H / 2) * (CAM_W / 2), 0.0f);
        fc1.out_act.assign(64, 0.0f);

        optimizer.configure(parameters(), learning_rate, 1e-4f);
        adam_step_count = 0;
    }

    std::vector<ValuePtr> parameters() const override {
        std::vector<ValuePtr> params;
        auto append = [&](const std::vector<ValuePtr>& v) {
            params.insert(params.end(), v.begin(), v.end());
        };
        append(conv1_layer.parameters());
        append(conv2_layer.parameters());
        append(conv3_layer.parameters());
        append(fc1_layer.parameters());
        append(fc2_layer.parameters());
        append(fc_out_layer.parameters());
        return params;
    }

    static std::array<float, AUX_DIM> build_aux_vector(const CityWorld& world, const RuleTelemetry& vis_cues) {
        std::array<float, AUX_DIM> aux{};
        aux[0] = (world.ego.active_intent == TurnIntent::LEFT)     ? 1.0f : 0.0f;
        aux[1] = (world.ego.active_intent == TurnIntent::STRAIGHT) ? 1.0f : 0.0f;
        aux[2] = (world.ego.active_intent == TurnIntent::RIGHT)    ? 1.0f : 0.0f;
        aux[3] = clampf(world.ego.speed / 12.0f, 0.0f, 1.5f);
        aux[4] = clampf(world.ego.steer_angle / 0.58f, -1.0f, 1.0f);

        float eff_cte = vis_cues.cross_track_error - vis_cues.evasive_lat_offset;
        aux[5] = clampf(-eff_cte * 0.45f - vis_cues.heading_error * 0.65f, -1.5f, 1.5f);
        Vec2 fwd = Vec2::from_angle(world.ego.heading);
        Vec2 left = fwd.perp_left();
        Vec2 to_la = vis_cues.lookahead_pt - world.ego.pos;
        float norm_lat = clampf(to_la.dot(left) / std::max(2.5f, vis_cues.lookahead_dist), -1.0f, 1.0f);
        aux[6] = norm_lat;
        aux[7] = clampf((vis_cues.target_speed - world.ego.speed) / 8.0f, -1.0f, 1.0f);
        return aux;
    }

    // Build autograd graph and run forward inference
    ValuePtr forward_graph(const float* raw_img_01, const float* aux_vec, bool need_input_pixel_grad, bool update_hud_views) {
        Tensor img_t({static_cast<size_t>(TOTAL_IMG_CHANNELS), static_cast<size_t>(CAM_H), static_cast<size_t>(CAM_W)});
        float* img_dst = img_t.data();
        for (int i = 0; i < TOTAL_IMG_FLOATS; ++i) {
            img_dst[i] = raw_img_01[i] * 2.0f - 1.0f;
        }

        Tensor aux_t({1, static_cast<size_t>(AUX_DIM)});
        std::memcpy(aux_t.data(), aux_vec, sizeof(float) * AUX_DIM);

        last_img_node = ValueNode::make(std::move(img_t), {}, "camera_input");
        auto aux_node = ValueNode::make(std::move(aux_t), {}, "aux_input");

        // 3-stage conv backbone
        auto c1 = EngineOps::max_pool2d(EngineOps::leaky_relu(conv1_layer.forward(last_img_node, need_input_pixel_grad)), 2);
        auto c2 = EngineOps::max_pool2d(EngineOps::leaky_relu(conv2_layer.forward(c1, true)), 2);
        auto c3 = EngineOps::max_pool2d(EngineOps::leaky_relu(conv3_layer.forward(c2, true)), 2);

        // Flatten visual features {20, 3, 4} -> {1, 240} and concatenate with auxiliary state {1, 8}
        auto vis_flat = EngineOps::flatten_to_row(c3);
        auto f1_in    = EngineOps::concat_cols(vis_flat, aux_node);
        auto h1       = EngineOps::leaky_relu(fc1_layer.forward(f1_in));

        // Skip-connect auxiliary state into the second dense layer
        auto f2_in    = EngineOps::concat_cols(h1, aux_node);
        auto h2       = EngineOps::leaky_relu(fc2_layer.forward(f2_in));
        last_out_node = EngineOps::tanh_op(fc_out_layer.forward(h2));

        predicted_steer = last_out_node->data.at_flat(0);
        predicted_accel = last_out_node->data.at_flat(1);

        if (update_hud_views) {
            std::memcpy(conv1.pool_out.data(), c1->data.data(), sizeof(float) * conv1.pool_out.size());
            std::memcpy(fc1.out_act.data(), h1->data.data(), sizeof(float) * fc1.out_act.size());
        }
        return last_out_node;
    }

    void forward(const float* raw_img_01, const float* aux_vec) {
        forward_graph(raw_img_01, aux_vec, true, true);
    }

    // Backpropagate |d(Action)/d(Pixel)| to visualize what regions of the camera feed drive the output
    void compute_saliency_maps() {
        if (!last_out_node || !last_img_node) return;

        Tensor seed({1, 2});
        seed.at_flat(0) = 1.0f;
        seed.at_flat(1) = 0.7f;
        last_out_node->backward(seed, true);

        const float* d_img = last_img_node->grad.data();
        auto normalize_cam_saliency = [&](int cam_idx, std::array<float, CAM_W * CAM_H>& out_sal) {
            float max_val = 1e-6f;
            const int base_ch = cam_idx * 3;
            for (int idx = 0; idx < CAM_W * CAM_H; ++idx) {
                float gx = d_img[(base_ch + 0) * CAM_W * CAM_H + idx];
                float gy = d_img[(base_ch + 1) * CAM_W * CAM_H + idx];
                float gz = d_img[(base_ch + 2) * CAM_W * CAM_H + idx];
                float mag = std::sqrt(gx * gx + gy * gy + gz * gz);
                out_sal[idx] = mag;
                if (mag > max_val) max_val = mag;
            }
            float inv_max = 1.0f / max_val;
            for (int idx = 0; idx < CAM_W * CAM_H; ++idx) {
                out_sal[idx] = clampf(out_sal[idx] * inv_max, 0.0f, 1.0f);
            }
        };

        normalize_cam_saliency(0, saliency_cam0);
        normalize_cam_saliency(1, saliency_cam1);

        // Clear parameter gradients accumulated during saliency attribution pass
        optimizer.zero_grad();
    }

    void zero_all_grads() {
        optimizer.zero_grad();
    }

    // Run forward + backward on one labeled sample and accumulate gradients
    float accumulate_sample_grad(const float* raw_img_01, const float* aux_vec,
                                 float target_steer, float target_accel, float sample_weight = 1.0f) {
        // Skip col2im on the first conv layer during weight updates since input pixel grads aren't needed here
        ValuePtr out = forward_graph(raw_img_01, aux_vec, false, false);

        float err_steer = predicted_steer - target_steer;
        float err_accel = predicted_accel - target_accel;
        float loss = 0.5f * (2.2f * err_steer * err_steer + 1.2f * err_accel * err_accel);

        Tensor d_out({1, 2});
        d_out.at_flat(0) = 2.2f * err_steer * sample_weight;
        d_out.at_flat(1) = 1.2f * err_accel * sample_weight;

        out->backward(d_out, true);
        return loss;
    }

    void apply_adamw_step(int batch_size) {
        optimizer.step(batch_size);
        adam_step_count = optimizer.step_count;
    }

    bool save_weights(const char* filepath) const {
        FILE* f = std::fopen(filepath, "wb");
        if (!f) return false;
        uint32_t magic = 0x455A4E50u; // "EZNP" (EngineZero Neural Policy)
        std::fwrite(&magic, sizeof(uint32_t), 1, f);
        std::fwrite(&adam_step_count, sizeof(uint64_t), 1, f);

        for (const auto& p : parameters()) {
            std::fwrite(p->data.data(), sizeof(float), p->data.size(), f);
        }
        std::fclose(f);
        return true;
    }

    bool load_weights(const char* filepath) {
        FILE* f = std::fopen(filepath, "rb");
        if (!f) return false;
        uint32_t magic = 0;
        if (std::fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x455A4E50u) {
            std::fclose(f);
            return false;
        }
        std::fread(&adam_step_count, sizeof(uint64_t), 1, f);
        optimizer.step_count = adam_step_count;

        for (auto& p : parameters()) {
            std::fread(p->data.data(), sizeof(float), p->data.size(), f);
        }
        std::fclose(f);
        return true;
    }

    size_t total_parameter_count() const {
        size_t total = 0;
        for (const auto& p : parameters()) {
            total += p->data.size();
        }
        return total;
    }
};

} // namespace sim
