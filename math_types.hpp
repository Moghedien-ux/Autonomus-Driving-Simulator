#pragma once
// Basic 2D/3D vectors, colors, cubic Beziers, and a small xorshift PRNG.
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sim {

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float DEG2RAD = PI / 180.0f;
constexpr float RAD2DEG = 180.0f / PI;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

inline float wrap_angle(float a) {
    while (a > PI)  a -= TWO_PI;
    while (a < -PI) a += TWO_PI;
    return a;
}

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2 operator/(float s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }

    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float cross(const Vec2& o) const { return x * o.y - y * o.x; }
    float len_sq() const { return x * x + y * y; }
    float len() const { return std::sqrt(x * x + y * y); }

    Vec2 normalized() const {
        float l = len();
        return l > 1e-6f ? Vec2{x / l, y / l} : Vec2{1.0f, 0.0f};
    }

    Vec2 perp_left() const { return {-y, x}; }
    Vec2 perp_right() const { return {y, -x}; }

    static Vec2 from_angle(float rad) {
        return {std::cos(rad), std::sin(rad)};
    }
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }
    float len() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const {
        float l = len();
        return l > 1e-6f ? Vec3{x / l, y / l, z / l} : Vec3{0.0f, 0.0f, 1.0f};
    }
};

struct Color3 {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    Color3() = default;
    constexpr Color3(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    Color3 operator*(float s) const {
        return {clampf(r * s, 0.0f, 1.0f), clampf(g * s, 0.0f, 1.0f), clampf(b * s, 0.0f, 1.0f)};
    }
    Color3 operator+(const Color3& o) const {
        return {clampf(r + o.r, 0.0f, 1.0f), clampf(g + o.g, 0.0f, 1.0f), clampf(b + o.b, 0.0f, 1.0f)};
    }
};

// Cubic Bezier curve used for road segments, intersection turns, and parking paths.
struct CubicBezier2D {
    Vec2 p0, p1, p2, p3;

    Vec2 eval(float t) const {
        float u = 1.0f - t;
        float tt = t * t;
        float uu = u * u;
        float uuu = uu * u;
        float ttt = tt * t;
        return p0 * uuu + p1 * (3.0f * uu * t) + p2 * (3.0f * u * tt) + p3 * ttt;
    }

    Vec2 tangent(float t) const {
        float u = 1.0f - t;
        return (p1 - p0) * (3.0f * u * u) +
               (p2 - p1) * (6.0f * u * t) +
               (p3 - p2) * (3.0f * t * t);
    }

    Vec2 second_deriv(float t) const {
        float u = 1.0f - t;
        return (p2 - p1 * 2.0f + p0) * (6.0f * u) +
               (p3 - p2 * 2.0f + p1) * (6.0f * t);
    }

    float curvature(float t) const {
        Vec2 d1 = tangent(t);
        Vec2 d2 = second_deriv(t);
        float num = d1.cross(d2);
        float den = std::pow(d1.len_sq(), 1.5f);
        return den > 1e-5f ? (num / den) : 0.0f;
    }
};

// Lightweight xorshift32 PRNG for deterministic procedural city generation.
struct FastRNG {
    uint32_t state = 0x1337BEEFu;

    explicit FastRNG(uint32_t seed = 0x1337BEEFu) : state(seed ? seed : 0x1337BEEFu) {}

    uint32_t next_u32() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    // [0, 1)
    float next_float() {
        return (next_u32() & 0x00FFFFFFu) / 16777216.0f;
    }

    // [lo, hi)
    float range(float lo, float hi) {
        return lo + (hi - lo) * next_float();
    }

    int range_int(int lo, int hi_inclusive) {
        if (hi_inclusive <= lo) return lo;
        return lo + static_cast<int>(next_u32() % static_cast<uint32_t>(hi_inclusive - lo + 1));
    }
};

} // namespace sim
