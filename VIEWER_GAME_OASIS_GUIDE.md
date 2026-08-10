# Game Oasis Player

This app lets you explore a generated, action-conditioned game world from one starting screenshot. It is intended for a model trained for the same game and visual style as that screenshot.

## Quick start

1. Install Python 3.10 or newer and the packages in `requirements.txt`:

   ```powershell
   py -3 -m pip install -r requirements.txt
   ```

2. Double-click `RUN_GAME_OASIS_PLAYER.bat`.
3. Open the **Action Player** tab.
4. Click **Add Downloaded Action Model (.zip or folder)** and choose the Kaggle ZIP download, or the extracted model folder.
5. Click **Choose Starting Image** and choose a clear in-game screenshot from the same game.
6. Click **Start Action Player**. Use the controls that were present in the model's recordings.

Format-v4 recordings keep arrow keys separate from WASD and also capture left,
middle, and right click; Enter; Shift/Ctrl/Alt; Tab/Escape; Q/E/R/F/Z/X/C/V;
and number keys 1–4. This supports rhythm lanes, click interactions, and
game-specific actions. Older datasets and models still load, but cannot learn
controls that were absent from their recordings.

Zero-input scenes remain animated by default so opponents, UI, particles, and
ambient effects do not become a still image. Adjust **Idle animation amount**, or
turn off **Keep no-input scenes animated** when a frozen idle view is preferable.

The first start takes longer because the model loads into memory. A CUDA-capable NVIDIA GPU is strongly recommended.

## What goes in the Kaggle download

Zip the complete trained action-model folder. The app supports a ZIP containing either the model files directly or one release folder containing them. The model needs this layout:

```text
My_Game_Action_Model.zip
  action_flow_model_info.json
  unet/
    config.json
    diffusion_pytorch_model.safetensors  (or .bin)
  ...any other model files produced during training
```

Do not distribute a base video model as the viewer download: the interactive player requires an Action Flow model with `action_flow_model_info.json`.

## Important expectation

This generates the next view from the current image and the selected controls. It is an interactive visual world model, not a connection to Roblox or any game server, and it does not reproduce game logic, multiplayer, inventories, or a persistent map. For best results, train each model on gameplay from one game and start it with a screenshot from that same game.
