# Universal Game Automation Bot — Documentation (English)

This document describes how to install and run the bot, how to configure an Android/Window backend (including Bluestacks), how to use the GUI, and how to author tasks. A full example based on the `treasure_dig` task is included.

## 1. Requirements & Quick install

Supported platforms: Windows and Linux. Python 3.9+ is required.

Recommended quick install (Linux / WSL / Windows PowerShell):

1. Clone the Repository
  ```bash
  git clone https://github.com/flyinghuman/gamebot.git
  cd gamebot
  ```

2. Create and activate a virtual environment:

   - Linux / macOS / WSL:

       python -m venv .venv
       source .venv/bin/activate

   - Windows (PowerShell):

       python -m venv .venv
       .\.venv\Scripts\Activate.ps1

3. Install Python requirements from the repository root:

       pip install -r requirements.txt

4. Configure ADB (if using Android backend) and place the active profile in `profiles/`.

Notes:
- Ensure `adb` (Android Platform Tools) is available on PATH when using the Android backend.
- On Windows, you may need to run PowerShell or the GUI as Administrator for some UI automation operations.

## 2. Installing the platform tools / ADB

- On Linux (Debian/Ubuntu):

      sudo apt update
      sudo apt install adb

- On Windows: Download the Android SDK Platform Tools from Google and either add `adb.exe` to your PATH or copy to a known folder. Test with:

      adb devices

If you intend to connect to an emulator or Android device make sure the device is listed.
- ADB for Windows can be downloaded here: https://dl.google.com/android/repository/platform-tools-latest-windows.zip

## 3. Bluestacks: top-level steps (enable ADB and set Performance to Maximum)

Bluestacks is a common Android emulator used on Windows. The bot can drive an emulator with ADB enabled. Top-level steps:

1. Install Bluestacks from the official website.
2. Open Bluestacks settings → Advanced (or Engine / Performance depending on version).
   - Set Performance / CPU / Memory to maximum or to the highest stable values available for your machine.
   - Apply and restart Bluestacks if prompted.
3. Enable ADB: In Bluestacks settings (Advanced / ADB), enable the ADB toggle to allow `adb` connections.
4. Connect from a host machine (if required): Some Bluestacks builds expose a TCP ADB endpoint (example: 127.0.0.1:5555). Use `adb connect <address>` if needed.
5. Verify with `adb devices` — Bluestacks or the emulator should appear.

Notes & tips:
- If Bluestacks doesn't expose ADB by default, consult the Bluestacks support docs for your specific version.
- Ensure virtualization (VT-x/AMD-V) is enabled in your BIOS/UEFI for best emulator performance.

## 4. Running the bot

From the repository root with the virtual environment active:

    python bot.py

Select the desired profile under Settings (or edit `profiles/active_profile.txt`) and then start the bot from the GUI.

## 5. GUI overview — each tab explained

The GUI exposes several tabs. Terminology here matches the strings used in the UI.

Monitor:
- Shows the live screenshot feed coming from the backend (ADB or window capture).
- Displays heatmaps and ROI overlays so you can verify template detections in near real time.
- Useful for tuning thresholds and template selection.

Dashboard:
- Shows runtime statistics and user-configurable task metrics.
- Recent Actions lists the last executed actions with timestamps and results.
- Use the metric selector and range dropdown to visualize time series metrics.

ROIs:
- Show Region of interests to speed up the template-matching Alogrithm
- You can create new ROIs or edit these

Templates:
- Manage template images referenced in profiles/.../`config.yaml`.
- Add up to 10 images per template key, reorder them, or replace them via screenshot capture with cropping them.

Tasks (Tasks):
- Create, edit and reorder tasks stored in the profile's `tasks.yaml`.
- Each task has an `id`, `name`, `trigger` and a set of `steps` describing the workflow.
- The built-in editor helps build conditional steps, loops, sleep, template-based actions and more.

Settings (Settings):
- Configure `config.yaml` options: thresholds, ROIs, GUI options, template directory and backend selection.
- Select the active profile and save configurations.

Log (Log):
- Structured log output for troubleshooting task execution, detection failures and runtime errors.

## 6. Task structure & how-to design tasks

Tasks live in the profile `tasks.yaml` file. High-level fields for a task:

- id: a unique identifier string (used when referencing the task programmatically).
- name: human-readable name.
- description: optional helpful text.
- enabled: boolean to enable/disable the task.
- trigger: scheduling info (interval, min/max seconds, run_at_start etc.)
- steps: an ordered list of step definitions executed when the task runs.
- stats: optional, defines numeric counters and series used for the Dashboard.

Common step types:
- tap_template: attempt to find a template and tap it (non-blocking, may be optional).
- wait_tap_template: waits until the template appears (blocking up to timeout) and taps it.
- sleep: pause execution for a given amount of seconds (or min/max randomized).
- loop: repeat an inner set of steps a number of times.
- if: conditional branching depending on a template detection, flag, or other condition.
- set_flag / set_detail / set_success: set internal state visible to later steps and the task result.
- call_task: invoke another task defined in the same `tasks.yaml`.
- press_back: send a back action to the backend (useful to close dialogs).

Design tips:
- Keep steps small and single-purpose — easier to test and reason about.
- Use `wait_tap_template` for flows that require the element to be present before continuing.
- Use `call_task` to encapsulate reusable routines (for example: close dialogs or stabilize home).
- Use `set_flag` to record intermediate detections and then branch with `if` on that flag.

## 7. Stats and Counters (Dashboard integration)

The profile GUI supports defining named counters/series that the Dashboard picks up. In `tasks.yaml` tasks can include a `stats` section where counters are declared. The exact GUI fields may vary by version, but a representative example:

Example `stats` snippet (YAML):

```yaml
stats:
  counters:
    - name: treasures_found
      label: Treasures Found
      source: action
      accumulate: sum
      unit: count

  series:
    - name: treasures_dug
      label: Treasures Dug
      source: action
      max_points: 200
      unit: count
```

How to use counters in tasks:
- When the task performs a measurable event (e.g., collects a treasure), the task should set a flag or call a step the GUI hooks into to increment the corresponding counter. The GUI and engine will update the Dashboard counters when such actions run.

If counters do not appear, ensure:
- The `stats` block is present under the task definition.
- The GUI profile is reloaded so the new counters are discovered.

## 8. Example: Treasure Hunt task (`treasure_dig`) — step-by-step

Below is a human-friendly explanation of the `treasure_dig` task included in the profile. It demonstrates common step types and how a full flow is constructed.

Purpose: Open the treasure view, detect a shared dig message, perform the dig sequence and collect the reward.

Key templates used:
- `dig_excavator_info_button` — small button on the bottom bar opening the dig view.
- `dig_share_info_message` — chat message pattern indicating a shared treasure dig exists.
- `dig_button`, `dig_do_dig_button`, `dig_send_troops_button` — sequence of action buttons in the dig UI.
- `dig_grab_gift_button` — finalize and collect the reward.

Typical flow (simplified):
1. tap_template: `dig_excavator_info_button` — attempt to open the dig UI. Uses a loose threshold to avoid missing the button.
2. sleep: 1–2s random pause to allow UI transition.
3. tap_template: `dig_share_info_message` (required: false) — not finding it is acceptable; if found, the task sets flags indicating a dig was detected.
4. if (flag check on dig_msg_found):
   - wait_tap_template: `dig_button` — ensures the dig button is present and taps it.
   - wait_tap_template: `dig_do_dig_button` — confirm dig action.
   - wait_tap_template: `dig_send_troops_button` — dig flow requires sending troops.
   - set_flag: `dig_treasure_dug = 1` and related series flags.
   - wait_tap_template: `dig_grab_gift_button` — collect reward and set `dig_reward_collected` flag.
5. call_task: `close_any_windows` — a helper routine that attempts to close open dialogs to return to a stable state.

Why this design works:
- Non-blocking initial taps avoid the task getting stuck when the UI is already in the desired state.
- `wait_tap_template` steps ensure the task waits for critical UI elements before continuing.
- Flags record the state of the flow and allow for clear `if` branching.

Customizing thresholds and ROI:
- Each template is defined in `config.yaml` with `threshold` and `roi` configuration. If detections are flaky, consider:
  - Lowering threshold slightly (e.g., 0.85 → 0.8) for tolerant matching.
  - Adjusting the ROI to include the smallest necessary portion of the screen.

## 9. Troubleshooting & tips

- If templates are not found:
  - Verify the template images in `profiles/<name>/templates/` match the device resolution.
  - Use the Monitor to locate the element manually and preview detections.

- If the bot fails to interact with a Windows game window:
  - Ensure `control_mode` in the profile `config.yaml` is set to `windows` and `windows_title` matches the window caption.

- If ADB devices do not show up:
  - Confirm `adb` is on PATH and `adb devices` lists the device.

## 10. Next steps & contributing

- Add new templates via the Templates tab and save the profile.
- Create small test tasks before composing larger flows. Use `run at start` for immediate testing.

---
If you want, make a donation to the developer to support further work and maintenance. Thank you for considering it!
BTC: bc1qydkjt2gfxr7uz4pt4jldpzwxqty8nc76vtv9ss
ETH: 0x5D0F170eBc8caC2db4F9477E26A4858142abDEEB
XRP: rUvTGRaqg9Q7DTJoMRGRVBZ1fpKaZQcx5a
DOGE: DRPr2oh2ynd4FJ66BrPFfskE1jpEWWnBpm
DASH: Xnzed2r417hGfgLG4DqCCTSNPci6FbvgHW
