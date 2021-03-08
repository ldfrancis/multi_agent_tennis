# environment
PLATFORM = "mac"  # either of mac, windows_x86, windows_x86_64,
# linux_x86, linux_x86_64, linux_x86_headless, linux_x86_64_headless,
ENV_FILE = {
    "mac": "Tennis.app",
    "windows_x86": "Tennis_Windows_x86/Tennis.exe",
    "windows_x86_64": "Tennis_Windows_x86_64/Tennis.exe",
    "linux_x86": "Tennis_Linux/Tennis.x86",
    "linux_x86_64": "/Tennis_Linux/Tennis.x86_64",
    "linux_x86_headless": "/Tennis_Linux_NoVis/Tennis.x86",
    "linux_x86_64_headless": "/Tennis_Linux_NoVis/Tennis.x86_64"
}[PLATFORM]
ENV_PATH = f"./tennis_env/{ENV_FILE}"
NUM_OBS = 24
NUM_ACT = 2
TARGET_SCORE = 0.5

# ddpg agent
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
ACTOR_HIDDEN_DIM = [256, 128]
CRITIC_HIDDEN_DIM = [256, 128]
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4




