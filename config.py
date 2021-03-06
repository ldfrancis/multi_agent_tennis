# environment
PLATFORM = "mac"  # either of mac, windows_x86, windows_x86_64,
# linux_x86, linux_x86_64, linux_x86_headless, linux_x86_64_headless,
ENV_FILE = {
    "mac": "Reacher.app",
    "windows_x86": "Reacher_Windows_x86/Reacher.exe",
    "windows_x86_64": "Reacher_Windows_x86_64/Reacher.exe",
    "linux_x86": "Reacher_Linux/Reacher.x86",
    "linux_x86_64": "/Reacher_Linux/Reacher.x86_64",
    "linux_x86_headless": "/Reacher_Linux_NoVis/Reacher.x86",
    "linux_x86_64_headless": "/Reacher_Linux_NoVis/Reacher.x86_64"
}[PLATFORM]
ENV_PATH = f"./reacher_env/{ENV_FILE}"
NUM_OBS = 33
NUM_ACT = 4
TARGET_SCORE = 30

# ppo agent
BATCH_SIZE = 64
GAMMA = 0.9
LAMBDA = 0.8
POLICY_HIDDEN_DIM = [64, 32]
CRITIC_HIDDEN_DIM = [64, 32]
MAX_LOG_STD = 0
MIN_LOG_STD = -20
ENTROPY_WEIGHT = 0.005
EPOCHS = 10
POLICY_LR = 5e-5
CRITIC_LR = 5e-5
CLIP_EPSILON = 0.2


