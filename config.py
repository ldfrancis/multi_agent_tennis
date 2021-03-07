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

# ippo agent
GAMMA = 1.0
LAMBDA = 1
POLICY_HIDDEN_DIM = [256, 256]
CRITIC_HIDDEN_DIM = [256, 256]
MAX_LOG_STD = 0
MIN_LOG_STD = -20
ENTROPY_WEIGHT = 1
EPOCHS = 10
POLICY_LR = 5e-5
CRITIC_LR = 5e-5
CLIP_EPSILON = 0.2


