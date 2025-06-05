// game.js
// -------------------------------------------------------------------
//  Canvas / socket setup
// -------------------------------------------------------------------
const canvas = document.getElementById('gameCanvas');
const ctx    = canvas.getContext('2d');
const socket = io();

console.log('[client] game.js loaded');

// -------------------------------------------------------------------
//  Local copy of game state (kept in sync with the server)
// -------------------------------------------------------------------
let gameState = {
  paddles: {
    left:  { y: 50, score: 0 },
    right: { y: 50, score: 0 }
  },
  ball:   { x: 400, y: 300, radius: 4},
  blocks: []
};

socket.on('connect', () => {
  console.log('[socket] connected, id =', socket.id);
});

socket.on('gameState', (state) => {
  gameState = state;
});

// -------------------------------------------------------------------
//  Keyboard handling
// -------------------------------------------------------------------
const keyState       = { ArrowUp: false, ArrowDown: false };
const PADDLE_SPEED   = 7;             // px per frame (manual mode)
const PADDLE_HEIGHT  = 100;           // for centering calc
const AUTO_SPEED     = 10;            // max px per frame in auto mode
let   autoTrack      = false;         // toggled by "K"

window.addEventListener('keydown', (e) => {
  const k = e.key;
  switch (k) {
    case 'ArrowUp':
    case 'ArrowDown':
      keyState[k] = true;
      console.log(`[key] keydown ${k}`);
      e.preventDefault();
      break;

    case 'r':
    case 'R':
      console.log('[key] "R" pressed – sending reset');
      socket.emit('reset', 'reset');
      e.preventDefault();
      break;

    case 'k':
    case 'K':
      autoTrack = !autoTrack;
      console.log(`[key] "K" pressed – auto-track ${autoTrack ? 'ENABLED' : 'DISABLED'}`);
      e.preventDefault();
      break;
  }
});

window.addEventListener('keyup', (e) => {
  const k = e.key;
  if (k === 'ArrowUp' || k === 'ArrowDown') {
    keyState[k] = false;
    console.log(`[key] keyup   ${k}`);
    e.preventDefault();
  }
});

// -------------------------------------------------------------------
//  Paddle movement helpers
// -------------------------------------------------------------------
function clampY(y) {        // keep paddle fully on-screen
  return Math.max(0, Math.min(500, y));
}

function sendRightPaddle(y) {
  gameState.paddles.right.y = y;            // optimistic local update
  socket.emit('paddleMove', { side: 'right', y });
}

function handleKeyboard() {                 // manual mode
  let newY = gameState.paddles.right.y;

  if (keyState.ArrowUp)   newY -= PADDLE_SPEED;
  if (keyState.ArrowDown) newY += PADDLE_SPEED;

  newY = clampY(newY);
  if (newY !== gameState.paddles.right.y) {
    console.log(`[paddle] manual move to y=${newY}`);
    sendRightPaddle(newY);
  }
}

function handleAutoTrack() {                // auto-track mode
  const targetY = clampY(gameState.ball.y - PADDLE_HEIGHT / 2);
  const currY   = gameState.paddles.right.y;
  let   diff    = targetY - currY;

  // limit speed so paddle doesn't teleport
  if (Math.abs(diff) > AUTO_SPEED) diff = Math.sign(diff) * AUTO_SPEED;

  const newY = currY + diff;
  if (newY !== currY) {
    console.log(`[paddle] auto-track move to y=${newY}`);
    sendRightPaddle(newY);
  }
}

// -------------------------------------------------------------------
//  Mouse control (unchanged)
// -------------------------------------------------------------------
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const y    = e.clientY - rect.top - 50;
  const x    = e.clientX - rect.left;
  const side = x < canvas.width / 2 ? 'left' : 'right';

  socket.emit('paddleMove', {
    side,
    y: clampY(y)
  });
});

// -------------------------------------------------------------------
//  Main game loop
// -------------------------------------------------------------------
function gameLoop() {
  if (autoTrack) {
    handleAutoTrack();
  } else {
    handleKeyboard();
  }
  draw();
  requestAnimationFrame(gameLoop);
}

// -------------------------------------------------------------------
//  Rendering
// -------------------------------------------------------------------
function draw() {
  // background
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // scores
  ctx.fillStyle = 'white';
  ctx.font = '30px Arial';
  ctx.fillText(gameState.paddles.left.score,  50, 30);
  ctx.fillText(gameState.reward, 50, 60);
  ctx.fillText(gameState.paddles.right.score, canvas.width - 80, 30);

  // paddles
  ctx.fillRect(10,                gameState.paddles.left.y,  10, 100);
  ctx.fillRect(canvas.width - 20, gameState.paddles.right.y, 10, 100);

  // ball
  ctx.beginPath();
  ctx.arc(gameState.ball.x, gameState.ball.y, gameState.ball.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.closePath();

  // blocks
  gameState.blocks.forEach(block => {
    if (block.active) {
      ctx.fillStyle = block.color;
      ctx.fillRect(block.x, block.y, 50, 40);
    }
  });
}

// -------------------------------------------------------------------
gameLoop();