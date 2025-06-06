"""
DQN‑controlled paddle client (single‑file version).

Quick start (CPU):
    pip install torch numpy "python-socketio[asyncio_client]"
    python dqn_paddle_client.py

CUDA users: follow the PyTorch install matrix, e.g.:
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import socketio

# ────────────────────────────────────────────────────────────────
# Game / environment constants
# ────────────────────────────────────────────────────────────────
SCREEN_HEIGHT = 500          # canvas height in pixels
MAX_PADDLE_SPEED = 8         # pixels we move paddle per action
PADDLE_OFFSET = 50           # Half of the paddle length

# ────────────────────────────────────────────────────────────────
# Reinforcement‑learning hyper‑parameters
# ────────────────────────────────────────────────────────────────
GAMMA = 0.95
ALPHA = 0.001                # Adam learning‑rate
EPSILON_START = 0.20
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05

BATCH_SIZE = 64
MEMORY_CAP = 5_000
TARGET_SYNC = 1_000          # gradient steps between policy → target copies

# ────────────────────────────────────────────────────────────────
# Checkpoint path
# ────────────────────────────────────────────────────────────────
CKPT_PATH = Path("dqn_paddle.pt")

# ────────────────────────────────────────────────────────────────
# Helper: convert server JSON → compact, normalised tensor
# ────────────────────────────────────────────────────────────────

def extract_state(packet: dict) -> torch.Tensor:
    """Return a (3,) float32 tensor scaled to roughly ±1."""
    paddle_y = packet["paddles"]["left"]["y"]
    ball = packet["ball"]
    rel_y = (ball["y"] - (paddle_y + PADDLE_OFFSET)) / (SCREEN_HEIGHT / 2)
    dy = ball["dy"]
    ball_x = ball["x"]
    ball_dx = ball["dx"]
    coming = 1.0 if ball["dx"] < 0 else 0.0
    return torch.tensor([rel_y, dy, ball_x, ball_dx], dtype=torch.float32)

# ────────────────────────────────────────────────────────────────
# DQN model
# ────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Q(s,·)
        return self.net(x)

# ────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ────────────────────────────────────────────────────────────────

def save_checkpoint(policy: DQN, optimiser: optim.Optimizer, eps: float, step: int) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optimiser.state_dict(),
            "eps": eps,
            "step": step,
        },
        CKPT_PATH,
    )
    print(f"[DQN] checkpoint saved → {CKPT_PATH.resolve()}")


def load_checkpoint(policy: DQN, optimiser: optim.Optimizer) -> Tuple[float, int]:
    if not CKPT_PATH.exists():
        print("[DQN] no checkpoint found — starting fresh.")
        return EPSILON_START, 0

    data = torch.load(CKPT_PATH, map_location="cpu")
    policy.load_state_dict(data["model"])
    try:
        optimiser.load_state_dict(data["optim"])
    except ValueError:
        print("[DQN] optimiser state mismatch; using new optimiser state.")
    print(f"[DQN] loaded checkpoint from {CKPT_PATH.resolve()}")
    return float(data.get("eps", EPSILON_START)), int(data.get("step", 0))

# ────────────────────────────────────────────────────────────────
# Global RL objects
# ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
punish_flag = False
reward_flag = False

optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)

epsilon, step_count = load_checkpoint(policy_net, optimizer)
replay: deque[tuple[torch.Tensor, int, float, torch.Tensor]] = deque(maxlen=MEMORY_CAP)

# ────────────────────────────────────────────────────────────────
# Socket.IO client & helpers
# ────────────────────────────────────────────────────────────────

sio = socketio.AsyncClient()

async def _send_paddle(side: str, new_y: int) -> None:
    await sio.emit("paddleMove", {"side": side, "y": new_y})

async def move_up(frame: dict, dy: int = MAX_PADDLE_SPEED) -> None:
    await _send_paddle("left", int(frame["paddles"]["left"]["y"]) + dy)

async def move_down(frame: dict, dy: int = MAX_PADDLE_SPEED) -> None:
    await _send_paddle("left", int(frame["paddles"]["left"]["y"]) - dy)

# ────────────────────────────────────────────────────────────────
# Training‑time state vars
# ────────────────────────────────────────────────────────────────
prev_state: torch.Tensor | None = None
prev_action: int | None = None
prev_left_score = 0

# ────────────────────────────────────────────────────────────────
# Socket.IO event handlers
# ────────────────────────────────────────────────────────────────

@sio.event
async def rewardHit(data):
    global reward_flag
    print("Rewarded for hit")
    reward_flag = True

@sio.event
async def punish(data):
    global punish_flag
    print("Punished for miss")
    punish_flag = True

@sio.event
async def gameState(data):
    global prev_state, prev_action, prev_left_score, epsilon, step_count, punish_flag, reward_flag

    state = extract_state(data).to(device)
    left_score = data["paddles"]["left"]["score"]

    # 1️⃣  Store previous transition
    if prev_state is not None:
        paddle_y = data["paddles"]["left"]["y"]
        ball_y = data["ball"]["y"]
        ball_x = data["ball"]["x"]
        paddle_lower = paddle_y + PADDLE_OFFSET * 2
        reward = 0
        if paddle_y <= ball_y <= paddle_lower:
            reward = 1
        else:
            reward = 1 / abs(paddle_y + PADDLE_OFFSET - ball_y) - 1
        reward += 100 * punish_flag + 10 * reward_flag
        print(f"{reward=}")
        to_send = {"reward": reward}
        await sio.emit("reward", to_send)
        replay.append((prev_state, prev_action, reward, state))
        punish_flag = False
        reward_flag = False

    # 2️⃣  ε‑greedy policy
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        with torch.no_grad():
            action = int(policy_net(state.unsqueeze(0)).argmax(dim=1).item())

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # 3️⃣  Execute action
    if action == 1:
        await move_up(data)
    elif action == 2:
        await move_down(data)
    # action 0 → stay

    # 4️⃣  Learn from replay buffer
    if len(replay) >= BATCH_SIZE:
        batch = random.sample(replay, BATCH_SIZE)
        s_batch = torch.stack([b[0] for b in batch]).to(device)
        a_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
        r_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        s2_batch = torch.stack([b[3] for b in batch]).to(device)

        q_sa = policy_net(s_batch).gather(1, a_batch).squeeze(1)
        with torch.no_grad():
            q_next = target_net(s2_batch).max(1)[0]
            targets = r_batch + GAMMA * q_next

        loss = nn.functional.mse_loss(q_sa, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1
        if step_count % TARGET_SYNC == 0:
            target_net.load_state_dict(policy_net.state_dict())
            save_checkpoint(policy_net, optimizer, epsilon, step_count)

    if not any(block['active'] for block in data['blocks']):
        await sio.emit('reset', 'reset')

    # 5️⃣  Prepare for next frame
    prev_state, prev_action, prev_left_score = state, action, left_score

@sio.event
async def reset(data):
    """Server sent a new‑episode reset → save the checkpoint."""
    save_checkpoint(policy_net, optimizer, epsilon, step_count)
    print("checkpoint saved on reset event.")

@sio.event
async def connect():
    print("Connected to server")

@sio.event
async def disconnect():
    save_checkpoint(policy_net, optimizer, epsilon, step_count)
    print("Disconnected — model saved")
    exit()

# ────────────────────────────────────────────────────────────────
# Async main entry‑point
# ────────────────────────────────────────────────────────────────

async def main() -> None:
    try:
        await sio.connect("http://localhost:3000")
        await sio.wait()
    finally:
        await sio.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
