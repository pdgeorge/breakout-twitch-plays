import asyncio
import socketio
import numpy as np
import random
from collections import defaultdict

ALPHA      = 0.7     # learning-rate (α)
GAMMA      = 0.95    # discount (γ)
EPSILON    = 0.2     # initial exploration rate (ε)
EPS_DECAY  = 0.9995  # multiply ε by this each time step
N_BINS_Y   = 12      # vertical resolution. Horizontal strips. Bigger = finer aiming but more table rows
N_BINS_DY  = 10      # velocity resolution. Bigger = agent distinguises subtle speed changes

SCREEN_HEIGHT          = 600
MAX_BALL_SPEED_Y       = 8
MAX_PADDLE_SPEED       = 4

REL_Y_RANGE_MIN        = -SCREEN_HEIGHT // 2
REL_Y_RANGE_MAX        = SCREEN_HEIGHT // 2

BALL_DY_RANGE_MIN      = -MAX_BALL_SPEED_Y
BALL_DY_RANGE_MAX      = MAX_BALL_SPEED_Y

def _bin(v, lo, hi, n_bins):
    """Clip b into [lo,hi] then return an int bin in [0, n_bins-1]."""
    v_clipped = max(min(v, hi), lo)
    return int((v_clipped - lo) / (hi - lo) * (n_bins - 1))

def extract_state(d: dict) -> tuple:
    """Convert raw JSON into a small, hashable RL state."""
    paddle_y    = d['paddles']['left']['y']
    ball        = d['ball']
    rel_y       = ball['y'] - paddle_y
    state = (_bin(rel_y, REL_Y_RANGE_MIN, REL_Y_RANGE_MAX, N_BINS_Y),
             _bin(ball['dy'], BALL_DY_RANGE_MIN, BALL_DY_RANGE_MAX, N_BINS_DY),
             1 if ball['dx'] < 0 else 0)
    return state

Q                 = defaultdict(lambda: np.zeros(3))  # 3 actions
prev_state        = None
prev_action       = None
prev_left_score   = 0
prev_right_score  = 0
global_epsilon    = EPSILON

sio = socketio.AsyncClient()

async def send_paddleMove(side, pos):
    client_msg={"side": side, "y": pos }
    await sio.emit('paddleMove', client_msg)

async def move_paddle_up(data, y: int = 1):
    new_y = int(data['paddles']['left']['y']) + y
    await send_paddleMove('left', new_y)

async def move_paddle_down(data, y: int = 1):
    new_y = int(data['paddles']['left']['y']) - y
    await send_paddleMove('left', new_y)

@sio.event
async def reset(data):
    pass
    # Do long training here

# Every time we receive an event of type 'gameState' this will be called
@sio.event
async def gameState(data):
    global prev_state, prev_action, prev_left_score, prev_right_score, global_epsilon

    # ------------------------------------------------------------------ update
    state = extract_state(data)

    if prev_state is not None:                        # skip very first frame
        # ----- reward since last frame
        left_score  = data['paddles']['left']['score']
        right_score = data['paddles']['right']['score']
        reward      = (left_score  - prev_left_score)

        # ----- TD(0) Q-update
        a             = prev_action
        best_next_Q   = np.max(Q[state])
        td_target     = reward + GAMMA * best_next_Q

        # "Short training"(tm)
        Q[prev_state][a] += ALPHA * (td_target - Q[prev_state][a])

        prev_left_score, prev_right_score = left_score, right_score

    # ------------------------------------------------------------------ choose
    if random.random() < global_epsilon:
        action = random.randint(0, 2)                 # explore
    else:
        action = int(np.argmax(Q[state]))             # exploit

    # linearly decay ε, but keep a floor so we never stop exploring
    global_epsilon = max(0.05, global_epsilon * EPS_DECAY)

    # ------------------------------------------------------------------ act
    if action == 1:
        await move_paddle_up(data, y=4)
    elif action == 2:
        await move_paddle_down(data, y=4)
    # action 0 = stay, so do nothing

    # keep a reference for the *next* step
    prev_state, prev_action = state, action

@sio.event
async def connect():
    print('Connected to server')

@sio.event
async def disconnect():
    print('Disconnected from server')

async def a_main():
    print("a_main")
    try:
        await sio.connect('http://localhost:3000')
        await sio.wait()
    except KeyboardInterrupt:
        await sio.disconnect()

def main():
    print("main")
    asyncio.run(a_main())

if __name__ == "__main__":
    main()