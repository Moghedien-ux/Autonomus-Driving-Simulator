#pragma once
// Vulkan 1.4 4x-MSAA renderer and 2D SDF UI pipeline.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include "scene_meshes.hpp"
#include "shaders/spirv_shaders.hpp"
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sim {

struct VertexUI {
    float x, y;
    float u, v;
    float r, g, b, a;
    float rx, ry, rw, rh; // SDF box center (rx, ry) and half-size (rw, rh)
    float mode, radius, border_w, tex_idx;
};

struct PushConst3D {
    float viewProj[16];
    float camPosAndTime[4];
};

struct PushConstUI {
    float screenSize[2];
    float pad[2];
};

struct GlyphMetrics {
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    float w = 0, h = 0, adv = 0;
};

// 4 anti-aliased Segoe UI font tiers baked into the 1024x512 R8 font atlas
enum FontTier : int {
    FONT_XS = 0,      // 14px Regular
    FONT_SM_BOLD = 1, // 15px Semibold
    FONT_LG_BOLD = 2, // 20px Bold
    FONT_SPEED = 3,   // 42px Bold (Digital Speedometer)
    FONT_TIER_COUNT = 4
};

class UIBuilder2D {
public:
    std::vector<VertexUI> verts;
    std::array<std::array<GlyphMetrics, 96>, FONT_TIER_COUNT> glyphs{};

    void clear() { verts.clear(); }

    void push_quad(float x0, float y0, float x1, float y1,
                   float u0, float v0, float u1, float v1,
                   Color3 col, float alpha,
                   float rx, float ry, float rw, float rh,
                   float mode, float radius, float border_w, float tex_idx) {
        VertexUI v00 = {x0, y0, u0, v0, col.r, col.g, col.b, alpha, rx, ry, rw, rh, mode, radius, border_w, tex_idx};
        VertexUI v10 = {x1, y0, u1, v0, col.r, col.g, col.b, alpha, rx, ry, rw, rh, mode, radius, border_w, tex_idx};
        VertexUI v11 = {x1, y1, u1, v1, col.r, col.g, col.b, alpha, rx, ry, rw, rh, mode, radius, border_w, tex_idx};
        VertexUI v01 = {x0, y1, u0, v1, col.r, col.g, col.b, alpha, rx, ry, rw, rh, mode, radius, border_w, tex_idx};
        verts.push_back(v00); verts.push_back(v10); verts.push_back(v11);
        verts.push_back(v00); verts.push_back(v11); verts.push_back(v01);
    }

    // Analytical Per-Pixel SDF Rounded Rectangle Fill
    void add_rounded_rect(float x0, float y0, float x1, float y1, float r, Color3 col, float alpha = 1.0f) {
        float cx = (x0 + x1) * 0.5f;
        float cy = (y0 + y1) * 0.5f;
        float hw = (x1 - x0) * 0.5f;
        float hh = (y1 - y0) * 0.5f;
        push_quad(x0 - 2.0f, y0 - 2.0f, x1 + 2.0f, y1 + 2.0f, 0, 0, 1, 1, col, alpha, cx, cy, hw, hh, 1.0f, r, 0.0f, 0.0f);
    }

    // Analytical Per-Pixel SDF Rounded Border Stroke
    void add_rounded_border(float x0, float y0, float x1, float y1, float r, Color3 col, float thickness = 1.2f, float alpha = 1.0f) {
        float cx = (x0 + x1) * 0.5f;
        float cy = (y0 + y1) * 0.5f;
        float hw = (x1 - x0) * 0.5f;
        float hh = (y1 - y0) * 0.5f;
        push_quad(x0 - 2.0f, y0 - 2.0f, x1 + 2.0f, y1 + 2.0f, 0, 0, 1, 1, col, alpha, cx, cy, hw, hh, 1.0f, r, thickness, 0.0f);
    }

    // Soft SDF Drop Shadow + Rounded Card Surface + Subtle Hairline Border
    void add_card(float x0, float y0, float x1, float y1, float r = 10.0f,
                  Color3 bg = {0.10f, 0.115f, 0.145f}, float alpha = 0.96f) {
        float cx = (x0 + x1) * 0.5f;
        float cy = (y0 + y1) * 0.5f;
        float hw = (x1 - x0) * 0.5f;
        float hh = (y1 - y0) * 0.5f;
        // Soft drop shadow (mode = 2)
        push_quad(x0 - 12.0f, y0 - 8.0f, x1 + 12.0f, y1 + 14.0f, 0, 0, 1, 1,
                  {0.02f, 0.03f, 0.05f}, 0.42f * alpha, cx, cy + 3.0f, hw, hh, 2.0f, r, 11.0f, 0.0f);
        // Card fill
        add_rounded_rect(x0, y0, x1, y1, r, bg, alpha);
        // Subtle 1.2px border
        add_rounded_border(x0, y0, x1, y1, r, {0.22f, 0.26f, 0.33f}, 1.2f, 0.72f * alpha);
    }

    // Analytical SDF Circular Ring (Steering Wheel & Friction Circle)
    void add_circle_ring(float cx, float cy, float radius, float thickness, Color3 col, float alpha = 1.0f) {
        float ext = radius + thickness + 3.0f;
        push_quad(cx - ext, cy - ext, cx + ext, cy + ext, 0, 0, 1, 1, col, alpha, cx, cy, ext, ext, 5.0f, radius, thickness, 0.0f);
    }

    // Anti-Aliased Thick Line Segment
    void add_line(float x0, float y0, float x1, float y1, float thickness, Color3 col, float alpha = 1.0f) {
        Vec2 d = {x1 - x0, y1 - y0};
        float len = d.len();
        if (len < 0.001f) return;
        Vec2 n = Vec2{-d.y / len, d.x / len} * (thickness * 0.5f);

        VertexUI v0 = {x0 - n.x, y0 - n.y, 0, 0, col.r, col.g, col.b, alpha, 0,0,0,0, 0,0,0,0};
        VertexUI v1 = {x1 - n.x, y1 - n.y, 0, 0, col.r, col.g, col.b, alpha, 0,0,0,0, 0,0,0,0};
        VertexUI v2 = {x1 + n.x, y1 + n.y, 0, 0, col.r, col.g, col.b, alpha, 0,0,0,0, 0,0,0,0};
        VertexUI v3 = {x0 + n.x, y0 + n.y, 0, 0, col.r, col.g, col.b, alpha, 0,0,0,0, 0,0,0,0};
        verts.push_back(v0); verts.push_back(v1); verts.push_back(v2);
        verts.push_back(v0); verts.push_back(v2); verts.push_back(v3);
    }

    // Vertical Gradient Quad (for area fill under loss chart)
    void add_vgrad_quad(float x0, float y_top0, float x1, float y_top1, float y_bottom,
                        Color3 col, float alpha_top, float alpha_bot) {
        VertexUI vt0 = {x0, y_top0,   0, 0, col.r, col.g, col.b, alpha_top, 0,0,0,0, 0,0,0,0};
        VertexUI vt1 = {x1, y_top1,   0, 0, col.r, col.g, col.b, alpha_top, 0,0,0,0, 0,0,0,0};
        VertexUI vb1 = {x1, y_bottom, 0, 0, col.r, col.g, col.b, alpha_bot, 0,0,0,0, 0,0,0,0};
        VertexUI vb0 = {x0, y_bottom, 0, 0, col.r, col.g, col.b, alpha_bot, 0,0,0,0, 0,0,0,0};
        verts.push_back(vt0); verts.push_back(vt1); verts.push_back(vb1);
        verts.push_back(vt0); verts.push_back(vb1); verts.push_back(vb0);
    }

    // Rounded Bilinear-Filtered Texture Widget (Saliency Maps & Conv2D Feature Maps)
    void add_textured_rect(float x0, float y0, float x1, float y1, float r, int tex_slot, float alpha = 1.0f) {
        float cx = (x0 + x1) * 0.5f;
        float cy = (y0 + y1) * 0.5f;
        float hw = (x1 - x0) * 0.5f;
        float hh = (y1 - y0) * 0.5f;
        push_quad(x0, y0, x1, y1, 0.0f, 0.0f, 1.0f, 1.0f, {1.0f, 1.0f, 1.0f}, alpha,
                  cx, cy, hw, hh, 4.0f, r, 0.0f, static_cast<float>(tex_slot));
    }

    // Exact pixel width of a string using baked TrueType glyph advance metrics
    float measure_text_width(FontTier tier, const char* text) const {
        if (!text) return 0.0f;
        float w = 0.0f;
        for (const char* p = text; *p; ++p) {
            unsigned char ch = static_cast<unsigned char>(*p);
            if (ch < 32 || ch >= 128) ch = '?';
            w += glyphs[tier][ch - 32].adv;
        }
        return w;
    }

    // Crisp Anti-Aliased Segoe UI Text Rendering from the Vulkan Font Atlas
    void add_text(FontTier tier, float x, float y_baseline, const char* text, Color3 col = {0.94f, 0.96f, 0.99f}, float alpha = 1.0f) {
        if (!text) return;
        float cur_x = std::floor(x);
        float base_y = std::floor(y_baseline);
        for (const char* p = text; *p; ++p) {
            unsigned char ch = static_cast<unsigned char>(*p);
            if (ch < 32 || ch >= 128) ch = '?';
            const GlyphMetrics& g = glyphs[tier][ch - 32];
            if (g.w > 0.0f && g.h > 0.0f) {
                float x0 = cur_x;
                float y0 = base_y - g.h * 0.80f;
                push_quad(x0, y0, x0 + g.w, y0 + g.h, g.u0, g.v0, g.u1, g.v1,
                          col, alpha, 0, 0, 0, 0, 3.0f, 0.0f, 0.0f, 0.0f);
            }
            cur_x += g.adv;
        }
    }

    // Horizontally centered text within [x0, x1]
    void add_text_centered(FontTier tier, float x0, float x1, float y_baseline, const char* text,
                           Color3 col = {0.94f, 0.96f, 0.99f}, float alpha = 1.0f) {
        float tw = measure_text_width(tier, text);
        float cx = x0 + ((x1 - x0) - tw) * 0.5f;
        add_text(tier, cx, y_baseline, text, col, alpha);
    }

    // Right-aligned text ending at x_right
    void add_text_right(FontTier tier, float x_right, float y_baseline, const char* text,
                        Color3 col = {0.94f, 0.96f, 0.99f}, float alpha = 1.0f) {
        float tw = measure_text_width(tier, text);
        add_text(tier, x_right - tw, y_baseline, text, col, alpha);
    }
};

struct VulkanTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
};

class VulkanRenderer {
public:
    static constexpr uint32_t MAX_3D_VERTS = 160000;
    static constexpr uint32_t MAX_UI_VERTS = 48000;
    static constexpr uint32_t FONT_ATLAS_W = 1024;
    static constexpr uint32_t FONT_ATLAS_H = 512;

    HWND hwnd = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    uint32_t graphics_queue_family = 0;
    char gpu_name[256] = "Unknown Vulkan GPU";
    VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_4_BIT;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchain_extent = {1600, 940};
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_views;
    std::vector<VkFramebuffer> framebuffers;

    // 4x MSAA Color & Depth Attachments
    VkImage msaa_color_image = VK_NULL_HANDLE;
    VkDeviceMemory msaa_color_mem = VK_NULL_HANDLE;
    VkImageView msaa_color_view = VK_NULL_HANDLE;

    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_mem = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;

    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout ui_desc_layout = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet ui_desc_set = VK_NULL_HANDLE;

    VkPipelineLayout layout_3d = VK_NULL_HANDLE;
    VkPipeline pipeline_3d = VK_NULL_HANDLE;
    VkPipelineLayout layout_ui = VK_NULL_HANDLE;
    VkPipeline pipeline_ui = VK_NULL_HANDLE;

    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkSemaphore sem_image_avail = VK_NULL_HANDLE;
    VkSemaphore sem_render_done = VK_NULL_HANDLE;
    VkFence fence_frame = VK_NULL_HANDLE;

    // Host-visible Dynamic Vertex Buffers (Main 3D, Camera Sensor 3D, UI Background, UI Foreground)
    VkBuffer vbo_3d_main = VK_NULL_HANDLE;
    VkDeviceMemory mem_3d_main = VK_NULL_HANDLE;
    void* mapped_3d_main = nullptr;

    VkBuffer vbo_3d_cam = VK_NULL_HANDLE;
    VkDeviceMemory mem_3d_cam = VK_NULL_HANDLE;
    void* mapped_3d_cam = nullptr;

    VkBuffer vbo_ui_bg = VK_NULL_HANDLE;
    VkDeviceMemory mem_ui_bg = VK_NULL_HANDLE;
    void* mapped_ui_bg = nullptr;

    VkBuffer vbo_ui_fg = VK_NULL_HANDLE;
    VkDeviceMemory mem_ui_fg = VK_NULL_HANDLE;
    void* mapped_ui_fg = nullptr;

    // Textures: [0] = Font Atlas, [1..2] = Dual Camera Saliency, [3..8] = 6 Conv2D Feature Maps
    std::array<VulkanTexture, 9> textures{};
    VkSampler linear_sampler = VK_NULL_HANDLE;

    // Host-visible staging buffer for per-frame Saliency & Feature Map texture uploads
    VkBuffer tex_staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory tex_staging_mem = VK_NULL_HANDLE;
    uint8_t* mapped_tex_staging = nullptr;

    bool init(HWND window_handle, int width, int height, UIBuilder2D& ui_builder) {
        hwnd = window_handle;
        swapchain_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

        create_instance_and_surface();
        pick_discrete_physical_device();
        create_logical_device();
        create_command_pool_and_sync();
        create_swapchain_and_renderpass();
        create_pipelines();
        create_vertex_buffers();
        create_textures_and_font_atlas(ui_builder);
        return true;
    }

    void recreate_swapchain(int width, int height) {
        if (width <= 0 || height <= 0) return;
        vkDeviceWaitIdle(device);
        cleanup_swapchain_objects();
        swapchain_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        create_swapchain_and_renderpass();
    }

    // Upload updated RGBA bytes for textures[1..8] (Saliency Maps & Conv2D Feature Maps)
    void record_dynamic_texture_uploads(VkCommandBuffer cb,
                                        const uint8_t* sal0_rgba, const uint8_t* sal1_rgba,
                                        const uint8_t fm_rgba[6][12 * 16 * 4]) {
        // Staging buffer layout:
        // [0 .. 3072): Sal0 (32x24x4 = 3072 bytes)
        // [3072 .. 6144): Sal1 (32x24x4 = 3072 bytes)
        // [6144 + f*768 .. ): 6 Feature Maps (16x12x4 = 768 bytes each)
        std::memcpy(mapped_tex_staging + 0,    sal0_rgba, 32 * 24 * 4);
        std::memcpy(mapped_tex_staging + 3072, sal1_rgba, 32 * 24 * 4);
        for (int f = 0; f < 6; ++f) {
            std::memcpy(mapped_tex_staging + 6144 + f * 768, fm_rgba[f], 16 * 12 * 4);
        }

        auto copy_region_to_tex = [&](int tex_idx, VkDeviceSize offset) {
            VulkanTexture& tex = textures[tex_idx];
            transition_image_layout(cb, tex.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {tex.width, tex.height, 1};
            vkCmdCopyBufferToImage(cb, tex_staging_buf, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            transition_image_layout(cb, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };

        copy_region_to_tex(1, 0);
        copy_region_to_tex(2, 3072);
        for (int f = 0; f < 6; ++f) {
            copy_region_to_tex(3 + f, 6144 + f * 768);
        }
    }

    void render_frame(const std::vector<Vertex3D>& verts_3d_main,
                      const std::vector<Vertex3D>& verts_3d_cam,
                      const std::vector<VertexUI>& verts_ui_bg,
                      const std::vector<VertexUI>& verts_ui_fg,
                      const PushConst3D& pc_main_3d,
                      const PushConst3D& pc_cam0_3d,
                      const PushConst3D& pc_cam1_3d,
                      VkRect2D cam0_rect,
                      VkRect2D cam1_rect,
                      bool draw_3d_sensor_views,
                      const uint8_t* sal0_rgba,
                      const uint8_t* sal1_rgba,
                      const uint8_t fm_rgba[6][12 * 16 * 4]) {
        vkWaitForFences(device, 1, &fence_frame, VK_TRUE, UINT64_MAX);

        uint32_t img_idx = 0;
        VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, sem_image_avail, VK_NULL_HANDLE, &img_idx);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
            return;
        }

        vkResetFences(device, 1, &fence_frame);

        // Copy dynamic geometry into persistent mapped vertex buffers
        uint32_t n_3d_main = std::min(static_cast<uint32_t>(verts_3d_main.size()), MAX_3D_VERTS);
        uint32_t n_3d_cam  = std::min(static_cast<uint32_t>(verts_3d_cam.size()),  MAX_3D_VERTS);
        uint32_t n_ui_bg   = std::min(static_cast<uint32_t>(verts_ui_bg.size()),   MAX_UI_VERTS);
        uint32_t n_ui_fg   = std::min(static_cast<uint32_t>(verts_ui_fg.size()),   MAX_UI_VERTS);

        if (n_3d_main > 0) std::memcpy(mapped_3d_main, verts_3d_main.data(), n_3d_main * sizeof(Vertex3D));
        if (n_3d_cam  > 0) std::memcpy(mapped_3d_cam,  verts_3d_cam.data(),  n_3d_cam  * sizeof(Vertex3D));
        if (n_ui_bg   > 0) std::memcpy(mapped_ui_bg,   verts_ui_bg.data(),   n_ui_bg   * sizeof(VertexUI));
        if (n_ui_fg   > 0) std::memcpy(mapped_ui_fg,   verts_ui_fg.data(),   n_ui_fg   * sizeof(VertexUI));

        vkResetCommandBuffer(cmd_buf, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_buf, &bi);

        // 1. Update dynamic Saliency & Conv2D Feature Map textures before RenderPass
        record_dynamic_texture_uploads(cmd_buf, sal0_rgba, sal1_rgba, fm_rgba);

        // 2. Begin 4x-MSAA RenderPass
        VkClearValue clears[2]{};
        clears[0].color = {{0.09f, 0.12f, 0.18f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpbi.renderPass = render_pass;
        rpbi.framebuffer = framebuffers[img_idx];
        rpbi.renderArea.offset = {0, 0};
        rpbi.renderArea.extent = swapchain_extent;
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clears;

        vkCmdBeginRenderPass(cmd_buf, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkDeviceSize zero_off = 0;
        const int main_vp_w = std::max(200, static_cast<int>(swapchain_extent.width) - 520);
        const int main_vp_h = static_cast<int>(swapchain_extent.height);

        // A. Draw Main 3D City Viewport (Full 32-bit Depth range [0.0, 1.0])
        if (n_3d_main > 0) {
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_3d);
            VkViewport vp = {0.0f, 0.0f, static_cast<float>(main_vp_w), static_cast<float>(main_vp_h), 0.0f, 1.0f};
            VkRect2D sc = {{0, 0}, {static_cast<uint32_t>(main_vp_w), static_cast<uint32_t>(main_vp_h)}};
            vkCmdSetViewport(cmd_buf, 0, 1, &vp);
            vkCmdSetScissor(cmd_buf, 0, 1, &sc);
            vkCmdPushConstants(cmd_buf, layout_3d, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConst3D), &pc_main_3d);
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vbo_3d_main, &zero_off);
            vkCmdDraw(cmd_buf, n_3d_main, 1, 0, 0);
        }

        // B. Draw 2D SDF UI Background Layer (Right Dock, Cards, Charts, Gauges)
        PushConstUI pc_ui = {{static_cast<float>(swapchain_extent.width), static_cast<float>(swapchain_extent.height)}, {0.0f, 0.0f}};
        VkViewport full_vp = {0.0f, 0.0f, static_cast<float>(swapchain_extent.width), static_cast<float>(swapchain_extent.height), 0.0f, 1.0f};
        VkRect2D full_sc = {{0, 0}, swapchain_extent};

        if (n_ui_bg > 0) {
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_ui);
            vkCmdSetViewport(cmd_buf, 0, 1, &full_vp);
            vkCmdSetScissor(cmd_buf, 0, 1, &full_sc);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_ui, 0, 1, &ui_desc_set, 0, nullptr);
            vkCmdPushConstants(cmd_buf, layout_ui, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstUI), &pc_ui);
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vbo_ui_bg, &zero_off);
            vkCmdDraw(cmd_buf, n_ui_bg, 1, 0, 0);
        }

        // C. Draw High-Res 3D Dual-Camera Sub-Viewports inside the Inspector Card (Full Depth [0.0, 1.0] with explicit Depth Clear)
        if (draw_3d_sensor_views && n_3d_cam > 0) {
            VkClearAttachment clear_depth{};
            clear_depth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            clear_depth.clearValue.depthStencil = {1.0f, 0};

            VkClearRect clear_rects[2] = {
                {cam0_rect, 0, 1},
                {cam1_rect, 0, 1}
            };
            vkCmdClearAttachments(cmd_buf, 1, &clear_depth, 2, clear_rects);

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_3d);
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vbo_3d_cam, &zero_off);

            // Camera 0 (Narrow 52 deg)
            VkViewport vp0 = {
                static_cast<float>(cam0_rect.offset.x), static_cast<float>(cam0_rect.offset.y),
                static_cast<float>(cam0_rect.extent.width), static_cast<float>(cam0_rect.extent.height),
                0.0f, 1.0f
            };
            vkCmdSetViewport(cmd_buf, 0, 1, &vp0);
            vkCmdSetScissor(cmd_buf, 0, 1, &cam0_rect);
            vkCmdPushConstants(cmd_buf, layout_3d, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConst3D), &pc_cam0_3d);
            vkCmdDraw(cmd_buf, n_3d_cam, 1, 0, 0);

            // Camera 1 (Wide 106 deg)
            VkViewport vp1 = {
                static_cast<float>(cam1_rect.offset.x), static_cast<float>(cam1_rect.offset.y),
                static_cast<float>(cam1_rect.extent.width), static_cast<float>(cam1_rect.extent.height),
                0.0f, 1.0f
            };
            vkCmdSetViewport(cmd_buf, 0, 1, &vp1);
            vkCmdSetScissor(cmd_buf, 0, 1, &cam1_rect);
            vkCmdPushConstants(cmd_buf, layout_3d, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConst3D), &pc_cam1_3d);
            vkCmdDraw(cmd_buf, n_3d_cam, 1, 0, 0);
        }

        // D. Draw 2D SDF UI Foreground Layer (Bilinear Saliency Overlays, Camera Borders & Segoe UI Text)
        if (n_ui_fg > 0) {
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_ui);
            vkCmdSetViewport(cmd_buf, 0, 1, &full_vp);
            vkCmdSetScissor(cmd_buf, 0, 1, &full_sc);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_ui, 0, 1, &ui_desc_set, 0, nullptr);
            vkCmdPushConstants(cmd_buf, layout_ui, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstUI), &pc_ui);
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vbo_ui_fg, &zero_off);
            vkCmdDraw(cmd_buf, n_ui_fg, 1, 0, 0);
        }

        vkCmdEndRenderPass(cmd_buf);
        vkEndCommandBuffer(cmd_buf);

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &sem_image_avail;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sem_render_done;
        vkQueueSubmit(graphics_queue, 1, &si, fence_frame);

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &sem_render_done;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &img_idx;
        vkQueuePresentKHR(graphics_queue, &pi);
    }

    void shutdown() {
        if (!device) return;
        vkDeviceWaitIdle(device);
        cleanup_swapchain_objects();
        if (linear_sampler) vkDestroySampler(device, linear_sampler, nullptr);
        for (auto& t : textures) {
            if (t.view) vkDestroyImageView(device, t.view, nullptr);
            if (t.image) vkDestroyImage(device, t.image, nullptr);
            if (t.memory) vkFreeMemory(device, t.memory, nullptr);
        }
        auto free_buf = [&](VkBuffer& b, VkDeviceMemory& m) {
            if (b) vkDestroyBuffer(device, b, nullptr);
            if (m) vkFreeMemory(device, m, nullptr);
        };
        free_buf(vbo_3d_main, mem_3d_main);
        free_buf(vbo_3d_cam,  mem_3d_cam);
        free_buf(vbo_ui_bg,   mem_ui_bg);
        free_buf(vbo_ui_fg,   mem_ui_fg);
        free_buf(tex_staging_buf, tex_staging_mem);

        if (pipeline_3d) vkDestroyPipeline(device, pipeline_3d, nullptr);
        if (layout_3d)   vkDestroyPipelineLayout(device, layout_3d, nullptr);
        if (pipeline_ui) vkDestroyPipeline(device, pipeline_ui, nullptr);
        if (layout_ui)   vkDestroyPipelineLayout(device, layout_ui, nullptr);
        if (desc_pool)   vkDestroyDescriptorPool(device, desc_pool, nullptr);
        if (ui_desc_layout) vkDestroyDescriptorSetLayout(device, ui_desc_layout, nullptr);
        if (sem_image_avail) vkDestroySemaphore(device, sem_image_avail, nullptr);
        if (sem_render_done) vkDestroySemaphore(device, sem_render_done, nullptr);
        if (fence_frame)     vkDestroyFence(device, fence_frame, nullptr);
        if (cmd_pool)        vkDestroyCommandPool(device, cmd_pool, nullptr);
        vkDestroyDevice(device, nullptr);
        if (surface)  vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
    }

private:
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_filter & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }

    void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                       VkBuffer& buf, VkDeviceMemory& mem, void** out_mapped = nullptr) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bci, nullptr, &buf);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
        vkAllocateMemory(device, &mai, nullptr, &mem);
        vkBindBufferMemory(device, buf, mem, 0);
        if (out_mapped) {
            vkMapMemory(device, mem, 0, size, 0, out_mapped);
        }
    }

    void create_image(uint32_t w, uint32_t h, VkFormat fmt, VkSampleCountFlagBits samples,
                      VkImageUsageFlags usage, VkImage& img, VkDeviceMemory& mem) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = fmt;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = usage;
        ici.samples = samples;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateImage(device, &ici, nullptr, &img);

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, img, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &mai, nullptr, &mem);
        vkBindImageMemory(device, img, mem, 0);
    }

    VkImageView create_image_view(VkImage img, VkFormat fmt, VkImageAspectFlags aspect) {
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = img;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = fmt;
        ivci.subresourceRange.aspectMask = aspect;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        vkCreateImageView(device, &ivci, nullptr, &view);
        return view;
    }

    void transition_image_layout(VkCommandBuffer cb, VkImage img, VkImageLayout old_layout, VkImageLayout new_layout) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = img;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }

        vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void create_instance_and_surface() {
        VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app_info.pApplicationName = "Autonomous Driving Simulator (EngineZero + Vulkan)";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "EngineZero";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app_info;
        ici.enabledExtensionCount = 2;
        ici.ppEnabledExtensionNames = exts;

        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        sci.hinstance = GetModuleHandleA(nullptr);
        sci.hwnd = hwnd;
        if (vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Win32 Vulkan surface");
        }
    }

    void pick_discrete_physical_device() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(instance, &count, devs.data());

        // Prefer Discrete GPU (NVIDIA GeForce RTX 4070 Laptop GPU)
        for (auto pd : devs) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device = pd;
                std::snprintf(gpu_name, sizeof(gpu_name), "%s", props.deviceName);
                break;
            }
        }
        if (!physical_device && !devs.empty()) {
            physical_device = devs[0];
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device, &props);
            std::snprintf(gpu_name, sizeof(gpu_name), "%s", props.deviceName);
        }

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qcount, qprops.data());

        for (uint32_t i = 0; i < qcount; ++i) {
            VkBool32 present_ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_ok);
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present_ok) {
                graphics_queue_family = i;
                break;
            }
        }
    }

    void create_logical_device() {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = graphics_queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        const char* dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkPhysicalDeviceFeatures features{};
        features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_exts;
        dci.pEnabledFeatures = &features;

        if (vkCreateDevice(physical_device, &dci, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan logical device");
        }
        vkGetDeviceQueue(device, graphics_queue_family, 0, &graphics_queue);
    }

    void create_command_pool_and_sync() {
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = graphics_queue_family;
        vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cbai, &cmd_buf);

        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device, &sci, nullptr, &sem_image_avail);
        vkCreateSemaphore(device, &sci, nullptr, &sem_render_done);

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fci, nullptr, &fence_frame);
    }

    void cleanup_swapchain_objects() {
        for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();
        for (auto iv : swapchain_views) vkDestroyImageView(device, iv, nullptr);
        swapchain_views.clear();

        if (msaa_color_view) vkDestroyImageView(device, msaa_color_view, nullptr);
        if (msaa_color_image) vkDestroyImage(device, msaa_color_image, nullptr);
        if (msaa_color_mem) vkFreeMemory(device, msaa_color_mem, nullptr);
        msaa_color_view = VK_NULL_HANDLE; msaa_color_image = VK_NULL_HANDLE; msaa_color_mem = VK_NULL_HANDLE;

        if (depth_view) vkDestroyImageView(device, depth_view, nullptr);
        if (depth_image) vkDestroyImage(device, depth_image, nullptr);
        if (depth_mem) vkFreeMemory(device, depth_mem, nullptr);
        depth_view = VK_NULL_HANDLE; depth_image = VK_NULL_HANDLE; depth_mem = VK_NULL_HANDLE;

        if (render_pass) vkDestroyRenderPass(device, render_pass, nullptr);
        render_pass = VK_NULL_HANDLE;

        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    void create_swapchain_and_renderpass() {
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps);
        if (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.width > 0) {
            swapchain_extent = caps.currentExtent;
        }

        uint32_t fmt_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &fmt_count, nullptr);
        std::vector<VkSurfaceFormatKHR> fmts(fmt_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &fmt_count, fmts.data());
        swapchain_format = fmts[0].format;
        VkColorSpaceKHR color_space = fmts[0].colorSpace;
        for (const auto& f : fmts) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
                swapchain_format = f.format;
                color_space = f.colorSpace;
                break;
            }
        }

        uint32_t img_count = std::max(2u, caps.minImageCount);
        if (caps.maxImageCount > 0 && img_count > caps.maxImageCount) img_count = caps.maxImageCount;

        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = surface;
        sci.minImageCount = img_count;
        sci.imageFormat = swapchain_format;
        sci.imageColorSpace = color_space;
        sci.imageExtent = swapchain_extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        sci.clipped = VK_TRUE;

        vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain);
        vkGetSwapchainImagesKHR(device, swapchain, &img_count, nullptr);
        swapchain_images.resize(img_count);
        vkGetSwapchainImagesKHR(device, swapchain, &img_count, swapchain_images.data());

        swapchain_views.resize(img_count);
        for (uint32_t i = 0; i < img_count; ++i) {
            swapchain_views[i] = create_image_view(swapchain_images[i], swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // Create 4x MSAA Color & Depth buffers
        create_image(swapchain_extent.width, swapchain_extent.height, swapchain_format, msaa_samples,
                     VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                     msaa_color_image, msaa_color_mem);
        msaa_color_view = create_image_view(msaa_color_image, swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT);

        create_image(swapchain_extent.width, swapchain_extent.height, VK_FORMAT_D32_SFLOAT, msaa_samples,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depth_image, depth_mem);
        depth_view = create_image_view(depth_image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

        // RenderPass: Attachment 0 = 4x MSAA Color, Attachment 1 = 4x MSAA Depth, Attachment 2 = 1x Resolve Present
        VkAttachmentDescription attachments[3]{};
        attachments[0].format = swapchain_format;
        attachments[0].samples = msaa_samples;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = msaa_samples;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        attachments[2].format = swapchain_format;
        attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference resolve_ref = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;
        subpass.pResolveAttachments = &resolve_ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpci.attachmentCount = 3;
        rpci.pAttachments = attachments;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &subpass;
        rpci.dependencyCount = 1;
        rpci.pDependencies = &dep;
        vkCreateRenderPass(device, &rpci, nullptr, &render_pass);

        framebuffers.resize(img_count);
        for (uint32_t i = 0; i < img_count; ++i) {
            VkImageView views[3] = {msaa_color_view, depth_view, swapchain_views[i]};
            VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fbci.renderPass = render_pass;
            fbci.attachmentCount = 3;
            fbci.pAttachments = views;
            fbci.width = swapchain_extent.width;
            fbci.height = swapchain_extent.height;
            fbci.layers = 1;
            vkCreateFramebuffer(device, &fbci, nullptr, &framebuffers[i]);
        }
    }

    VkShaderModule create_shader_module(const uint8_t* code, size_t size) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = size;
        smci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule mod = VK_NULL_HANDLE;
        vkCreateShaderModule(device, &smci, nullptr, &mod);
        return mod;
    }

    void create_pipelines() {
        // 1. Descriptor Set Layout for 2D UI Shader (9 combined image samplers)
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 9;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 1;
        dslci.pBindings = &binding;
        vkCreateDescriptorSetLayout(device, &dslci, nullptr, &ui_desc_layout);

        // 2. Pipeline Layout 3D
        VkPushConstantRange pcr_3d{};
        pcr_3d.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr_3d.offset = 0;
        pcr_3d.size = sizeof(PushConst3D);

        VkPipelineLayoutCreateInfo plci_3d{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci_3d.pushConstantRangeCount = 1;
        plci_3d.pPushConstantRanges = &pcr_3d;
        vkCreatePipelineLayout(device, &plci_3d, nullptr, &layout_3d);

        // 3. Pipeline Layout UI
        VkPushConstantRange pcr_ui{};
        pcr_ui.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr_ui.offset = 0;
        pcr_ui.size = sizeof(PushConstUI);

        VkPipelineLayoutCreateInfo plci_ui{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci_ui.setLayoutCount = 1;
        plci_ui.pSetLayouts = &ui_desc_layout;
        plci_ui.pushConstantRangeCount = 1;
        plci_ui.pPushConstantRanges = &pcr_ui;
        vkCreatePipelineLayout(device, &plci_ui, nullptr, &layout_ui);

        // Shared pipeline states
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp_state.viewportCount = 1;
        vp_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = msaa_samples;

        VkPipelineColorBlendAttachmentState blend_att{};
        blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_att.blendEnable = VK_TRUE;
        blend_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_att.colorBlendOp = VK_BLEND_OP_ADD;
        blend_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_att.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &blend_att;

        VkDynamicState dyn_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        // Build 3D Graphics Pipeline
        VkShaderModule mod_3d_v = create_shader_module(SPV_SCENE3D_VERT, sizeof(SPV_SCENE3D_VERT));
        VkShaderModule mod_3d_f = create_shader_module(SPV_SCENE3D_FRAG, sizeof(SPV_SCENE3D_FRAG));

        VkPipelineShaderStageCreateInfo stages_3d[2]{};
        stages_3d[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages_3d[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages_3d[0].module = mod_3d_v;
        stages_3d[0].pName = "main";
        stages_3d[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages_3d[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages_3d[1].module = mod_3d_f;
        stages_3d[1].pName = "main";

        VkVertexInputBindingDescription bind_3d = {0, sizeof(Vertex3D), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs_3d[5] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex3D, px)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex3D, nx)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex3D, r)},
            {3, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex3D, u)},
            {4, 0, VK_FORMAT_R32_SFLOAT,          offsetof(Vertex3D, mat_type)}
        };
        VkPipelineVertexInputStateCreateInfo vi_3d{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi_3d.vertexBindingDescriptionCount = 1;
        vi_3d.pVertexBindingDescriptions = &bind_3d;
        vi_3d.vertexAttributeDescriptionCount = 5;
        vi_3d.pVertexAttributeDescriptions = attrs_3d;

        VkPipelineDepthStencilStateCreateInfo ds_3d{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_3d.depthTestEnable = VK_TRUE;
        ds_3d.depthWriteEnable = VK_TRUE;
        ds_3d.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkGraphicsPipelineCreateInfo gp_3d{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp_3d.stageCount = 2;
        gp_3d.pStages = stages_3d;
        gp_3d.pVertexInputState = &vi_3d;
        gp_3d.pInputAssemblyState = &ia;
        gp_3d.pViewportState = &vp_state;
        gp_3d.pRasterizationState = &rast;
        gp_3d.pMultisampleState = &ms;
        gp_3d.pDepthStencilState = &ds_3d;
        gp_3d.pColorBlendState = &blend;
        gp_3d.pDynamicState = &dyn;
        gp_3d.layout = layout_3d;
        gp_3d.renderPass = render_pass;
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_3d, nullptr, &pipeline_3d);

        vkDestroyShaderModule(device, mod_3d_v, nullptr);
        vkDestroyShaderModule(device, mod_3d_f, nullptr);

        // Build 2D SDF UI Graphics Pipeline
        VkShaderModule mod_ui_v = create_shader_module(SPV_UI2D_VERT, sizeof(SPV_UI2D_VERT));
        VkShaderModule mod_ui_f = create_shader_module(SPV_UI2D_FRAG, sizeof(SPV_UI2D_FRAG));

        VkPipelineShaderStageCreateInfo stages_ui[2]{};
        stages_ui[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages_ui[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages_ui[0].module = mod_ui_v;
        stages_ui[0].pName = "main";
        stages_ui[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages_ui[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages_ui[1].module = mod_ui_f;
        stages_ui[1].pName = "main";

        VkVertexInputBindingDescription bind_ui = {0, sizeof(VertexUI), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs_ui[5] = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(VertexUI, x)},
            {1, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(VertexUI, u)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexUI, r)},
            {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexUI, rx)},
            {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexUI, mode)}
        };
        VkPipelineVertexInputStateCreateInfo vi_ui{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi_ui.vertexBindingDescriptionCount = 1;
        vi_ui.pVertexBindingDescriptions = &bind_ui;
        vi_ui.vertexAttributeDescriptionCount = 5;
        vi_ui.pVertexAttributeDescriptions = attrs_ui;

        VkPipelineDepthStencilStateCreateInfo ds_ui{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_ui.depthTestEnable = VK_FALSE;
        ds_ui.depthWriteEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo gp_ui{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp_ui.stageCount = 2;
        gp_ui.pStages = stages_ui;
        gp_ui.pVertexInputState = &vi_ui;
        gp_ui.pInputAssemblyState = &ia;
        gp_ui.pViewportState = &vp_state;
        gp_ui.pRasterizationState = &rast;
        gp_ui.pMultisampleState = &ms;
        gp_ui.pDepthStencilState = &ds_ui;
        gp_ui.pColorBlendState = &blend;
        gp_ui.pDynamicState = &dyn;
        gp_ui.layout = layout_ui;
        gp_ui.renderPass = render_pass;
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ui, nullptr, &pipeline_ui);

        vkDestroyShaderModule(device, mod_ui_v, nullptr);
        vkDestroyShaderModule(device, mod_ui_f, nullptr);
    }

    void create_vertex_buffers() {
        VkMemoryPropertyFlags host_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        create_buffer(MAX_3D_VERTS * sizeof(Vertex3D), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_props, vbo_3d_main, mem_3d_main, &mapped_3d_main);
        create_buffer(MAX_3D_VERTS * sizeof(Vertex3D), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_props, vbo_3d_cam,  mem_3d_cam,  &mapped_3d_cam);
        create_buffer(MAX_UI_VERTS * sizeof(VertexUI), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_props, vbo_ui_bg,   mem_ui_bg,   &mapped_ui_bg);
        create_buffer(MAX_UI_VERTS * sizeof(VertexUI), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_props, vbo_ui_fg,   mem_ui_fg,   &mapped_ui_fg);
    }

    void create_textures_and_font_atlas(UIBuilder2D& ui_builder) {
        // 1. Bilinear Sampler
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &sci, nullptr, &linear_sampler);

        // 2. Rasterize Anti-Aliased Segoe UI Font Atlas (1024x512 R8) using Win32 DIBSection
        std::vector<uint8_t> atlas_r8(FONT_ATLAS_W * FONT_ATLAS_H, 0);
        {
            BITMAPINFO bmi{};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = FONT_ATLAS_W;
            bmi.bmiHeader.biHeight = -static_cast<LONG>(FONT_ATLAS_H); // top-down
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            void* dib_bits = nullptr;
            HDC mem_dc = CreateCompatibleDC(nullptr);
            HBITMAP hbmp = CreateDIBSection(mem_dc, &bmi, DIB_RGB_COLORS, &dib_bits, nullptr, 0);
            SelectObject(mem_dc, hbmp);
            SetBkColor(mem_dc, RGB(0, 0, 0));
            SetTextColor(mem_dc, RGB(255, 255, 255));
            SetBkMode(mem_dc, TRANSPARENT);

            int heights[FONT_TIER_COUNT] = {13, 14, 19, 40};
            int weights[FONT_TIER_COUNT] = {FW_NORMAL, FW_SEMIBOLD, FW_BOLD, FW_BOLD};
            int pen_x = 4;
            int pen_y = 4;
            int row_h = 0;

            for (int tier = 0; tier < FONT_TIER_COUNT; ++tier) {
                HFONT hfont = CreateFontA(
                    -heights[tier], 0, 0, 0, weights[tier], FALSE, FALSE, FALSE,
                    ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
                    ANTIALIASED_QUALITY, FF_DONTCARE | VARIABLE_PITCH, "Segoe UI"
                );
                HFONT old_f = static_cast<HFONT>(SelectObject(mem_dc, hfont));

                for (int c = 32; c < 128; ++c) {
                    char ch = static_cast<char>(c);
                    SIZE sz{};
                    GetTextExtentPoint32A(mem_dc, &ch, 1, &sz);
                    ABC abc{};
                    float exact_adv = static_cast<float>(sz.cx);
                    if (GetCharABCWidthsA(mem_dc, c, c, &abc)) {
                        exact_adv = static_cast<float>(abc.abcA + static_cast<int>(abc.abcB) + abc.abcC);
                    }
                    int gw = std::max(1, static_cast<int>(sz.cx) + 2);
                    int gh = std::max(1, static_cast<int>(sz.cy));

                    if (pen_x + gw + 4 >= static_cast<int>(FONT_ATLAS_W)) {
                        pen_x = 4;
                        pen_y += row_h + 4;
                        row_h = 0;
                    }
                    TextOutA(mem_dc, pen_x, pen_y, &ch, 1);

                    GlyphMetrics gm;
                    gm.u0 = static_cast<float>(pen_x) / FONT_ATLAS_W;
                    gm.v0 = static_cast<float>(pen_y) / FONT_ATLAS_H;
                    gm.u1 = static_cast<float>(pen_x + sz.cx) / FONT_ATLAS_W;
                    gm.v1 = static_cast<float>(pen_y + gh) / FONT_ATLAS_H;
                    gm.w  = static_cast<float>(sz.cx);
                    gm.h  = static_cast<float>(gh);
                    gm.adv = std::max(3.0f, exact_adv);
                    ui_builder.glyphs[tier][c - 32] = gm;

                    pen_x += gw + 4;
                    row_h = std::max(row_h, gh);
                }
                SelectObject(mem_dc, old_f);
                DeleteObject(hfont);
            }
            GdiFlush();

            const uint32_t* pixels = static_cast<const uint32_t*>(dib_bits);
            for (size_t i = 0; i < FONT_ATLAS_W * FONT_ATLAS_H; ++i) {
                uint32_t px = pixels[i];
                uint32_t r = (px >> 16) & 0xFF;
                uint32_t g = (px >> 8) & 0xFF;
                uint32_t b = px & 0xFF;
                atlas_r8[i] = static_cast<uint8_t>((r + g + b) / 3);
            }
            DeleteObject(hbmp);
            DeleteDC(mem_dc);
        }

        // 3. Allocate Vulkan Textures [0..8]
        auto init_tex = [&](int idx, uint32_t w, uint32_t h, VkFormat fmt) {
            textures[idx].width = w;
            textures[idx].height = h;
            textures[idx].format = fmt;
            create_image(w, h, fmt, VK_SAMPLE_COUNT_1_BIT,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         textures[idx].image, textures[idx].memory);
            textures[idx].view = create_image_view(textures[idx].image, fmt, VK_IMAGE_ASPECT_COLOR_BIT);
        };

        init_tex(0, FONT_ATLAS_W, FONT_ATLAS_H, VK_FORMAT_R8_UNORM);
        init_tex(1, CAM_W, CAM_H, VK_FORMAT_R8G8B8A8_UNORM);
        init_tex(2, CAM_W, CAM_H, VK_FORMAT_R8G8B8A8_UNORM);
        for (int f = 0; f < 6; ++f) {
            init_tex(3 + f, CAM_W / 2, CAM_H / 2, VK_FORMAT_R8G8B8A8_UNORM);
        }

        // 4. Create dynamic staging buffer (16 KB) for per-frame Saliency & Feature Map updates
        void* st_ptr = nullptr;
        create_buffer(16384, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      tex_staging_buf, tex_staging_mem, &st_ptr);
        mapped_tex_staging = static_cast<uint8_t*>(st_ptr);

        // 5. Upload Font Atlas & transition all 9 textures to SHADER_READ_ONLY_OPTIMAL
        VkBuffer font_stage_buf = VK_NULL_HANDLE;
        VkDeviceMemory font_stage_mem = VK_NULL_HANDLE;
        void* font_mapped = nullptr;
        create_buffer(atlas_r8.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      font_stage_buf, font_stage_mem, &font_mapped);
        std::memcpy(font_mapped, atlas_r8.data(), atlas_r8.size());

        vkResetCommandBuffer(cmd_buf, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_buf, &bi);

        transition_image_layout(cmd_buf, textures[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy font_copy{};
        font_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        font_copy.imageSubresource.layerCount = 1;
        font_copy.imageExtent = {FONT_ATLAS_W, FONT_ATLAS_H, 1};
        vkCmdCopyBufferToImage(cmd_buf, font_stage_buf, textures[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &font_copy);
        transition_image_layout(cmd_buf, textures[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        for (int i = 1; i < 9; ++i) {
            transition_image_layout(cmd_buf, textures[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        vkEndCommandBuffer(cmd_buf);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue);

        vkDestroyBuffer(device, font_stage_buf, nullptr);
        vkFreeMemory(device, font_stage_mem, nullptr);

        // 6. Create Descriptor Pool & bind all 9 textures to ui_desc_set
        VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool);

        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &ui_desc_layout;
        vkAllocateDescriptorSets(device, &dsai, &ui_desc_set);

        VkDescriptorImageInfo img_infos[9]{};
        for (int i = 0; i < 9; ++i) {
            img_infos[i].sampler = linear_sampler;
            img_infos[i].imageView = textures[i].view;
            img_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet = ui_desc_set;
        wds.dstBinding = 0;
        wds.descriptorCount = 9;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = img_infos;
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }
};

} // namespace sim
