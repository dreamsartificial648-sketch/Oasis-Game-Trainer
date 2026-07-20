ROBLOX OASIS - NOISE TREADMILL

1. Put these files beside your existing flow_matching_app(12).py:
   - roblox_oasis_noise_treadmill.py
   - run_roblox_oasis.bat

2. Run run_roblox_oasis.bat once. It creates:
   1 VIDEO MODEL IN HERE

3. Copy ONE complete trained video-model folder into that folder.
   The model folder must contain:
   - flow_video_model_info.json
   - unet/config.json
   - the remaining UNet model files

4. Start the app, choose an optional starting screenshot, and press Reset World.

5. Click Select Player on Preview, then click the center of the Roblox player.
   Adjust Player Mask Radius and Mask Feather until the blue feathered overlay
   covers the player without covering too much of the environment.

6. Press Start Oasis and use:
   W = forward
   S = backward
   A = turn left
   D = turn right
   Space = jump

Recommended first settings for an RTX 3060 12 GB:
- Resolution: whatever the video model was trained at
- Interactive flow steps: 2-4
- ODE method: Euler
- Movement strength: 0.012-0.022
- Directional noise: 0.20-0.40
- Old-frame feedback: 0.12-0.25
- Player preservation: 0.90-0.98

Notes:
- This is a generated-world illusion, not true action-conditioned gameplay.
- The protected player is intentionally stabilized and may look nearly frozen.
- If the world mutates too much, lower Directional Noise and raise Old-frame Feedback.
- If movement barely appears, raise Movement Strength gradually.
