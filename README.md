# Autonomous Driving Simulator (EngineZero + Vulkan 1.4)

A self-contained 3D autonomous driving simulator and online imitation learning testbed written in C++20 with a custom reverse-mode autograd engine (**EngineZero**) and a 4x-MSAA **Vulkan 1.4** renderer.

## Overview

This project combines a rule-based urban motion planner (`v11 Rules`), an end-to-end vision + kinematic policy network (`v12 Neural`), and an online **DAgger** (Dataset Aggregation) trainer running live at 60 FPS:

- **EngineZero Autograd & Neural Policy (`engine_zero/`, `cnn_engine_v12.hpp`)**
  - Custom C++20 tensor library and dynamic computation graph with reverse-mode automatic differentiation (`Tensor`, `Autograd`, `Conv2d`, `Linear`, `AdamW`).
  - Dual synthetic windshield cameras (`32x24` narrow forward `52°` FOV + wide surround `108°` FOV) fused with an 18-dimensional ego/navigation/range vector.
  - Real-time input-Jacobian saliency backprop (`|dOut / dPixel|`) and Conv2D spatial feature map visualization.

- **Urban World & Traffic Simulation (`city_world.hpp`)**
  - Procedural multi-intersection city grid featuring multi-lane boulevards, one-way streets, 45° diagonal avenues, and a central 2-lane roundabout plaza.
  - Diverse intersection control: traffic lights, 3D animated traffic police officers with whistle/hand-gesture phases, 4-way stop signs, and yield signs.
  - Dynamic traffic actors including transit buses stopping at designated curb zones, hazard-stopped vehicles, bicycles, crosswalk pedestrians, and red-light runners.
  - 360° line-of-sight (LOS) raycast occlusion against city buildings and vehicles.

- **Rule-Based Motion Planner & Autopark (`rule_driver_v11.hpp`)**
  - Pure Pursuit + Stanley path tracking, Intelligent Driver Model (IDM) longitudinal control, corner-grip speed profiling, yellow-light dilemma zone evaluation, and unprotected left-turn yielding.
  - Safe two-way and multi-lane overtaking with oncoming traffic gap checks, forward conflict horizon checks, and standstill deadlock tie-breakers.
  - Multi-layout **Autopark** supporting curbside lateral (parallel) stalls—including direct straight-in entry for open Lane-1 lateral spots—as well as off-street 90° reverse-in, 90° front-in, and 45° angled stalls using world-anchored Cubic Bezier trajectories.

- **Vulkan 1.4 Renderer & Inspector UI (`vulkan_renderer.hpp`, `scene_meshes.hpp`, `main.cpp`)**
  - Hardware 4x-MSAA 3D viewport with procedural city geometry, perception bounding boxes, trajectory ribbons, and a 360° sensor ring.
  - 520px right-hand neural inspector dock rendering live dual-camera feeds, saliency heatmaps, Conv2D feature maps, steering/grip gauges, and the DAgger loss curve.

## Controls

| Key / Input | Action |
| :--- | :--- |
| `1` | Switch to **v11 Rules** (heuristic planner) |
| `2` | Switch to **DAgger Train** (online imitation learning with safety supervisor) |
| `3` | Switch to **v12 Neural** (end-to-end neural network control) |
| `4` | Switch to **Manual Drive** (`W`/`A`/`S`/`D` or Arrow keys) |
| `P` | Toggle **Autopark** search / unpark from stall |
| `T` | Run a fast background batch of 300 DAgger training epochs |
| `E` | Spawn a test traffic scenario ahead of the ego vehicle |
| `R` | Reset CNN weights and clear the DAgger replay buffer |
| `N` | Generate a new procedural city layout |
| `V` | Cycle camera view (`Chase 3D`, `Orbit 3D`, `City Map`) |
| `S` | Cycle camera saliency overlay (`Blend`, `Heatmap`, `Raw RGB`) |
| `B` | Toggle 3D perception bounding boxes |
| `Space` | Pause / resume simulation |
| `Right Mouse Drag` / `Wheel` | Orbit camera yaw/pitch and zoom |

## Requirements & Building

- **OS**: Windows 10 / 11 (x64)
- **Compiler**: MSVC (Visual Studio 2022 / Build Tools with C++20 support)
- **Graphics SDK**: [LunarG Vulkan SDK](https://vulkan.lunarg.com/) (update `VK_SDK` in `build_and_run.bat` if your SDK path differs)

To compile and launch the simulator:

```bat
build_and_run.bat
```

To run the headless verification suite (validating perception, traffic rules, overtaking, autopark, and DAgger convergence):

```bat
build_and_run.bat --headless-test
```

## License

Released under the [MIT License](LICENSE).
