// server.js
// ------------------------------------------------------------------
//  Simple Breakout / Pong hybrid server using Socket.IO
// ------------------------------------------------------------------
const express = require('express');
const http    = require('http');
const { Server } = require('socket.io');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server);

app.use(express.static('public'));

// ------------------------------------------------------------------
//  Game constants
// ------------------------------------------------------------------
const FPS           = 60;   // frames per second
const BLOCK_W       = 50;
const BLOCK_H       = 40;
const BALL_RAD      = 4;
const CANVAS_W      = 800;
const CANVAS_H      = 600;
const BRICK_GAP     = 5;    // px between bricks

// ------------------------------------------------------------------
//  Game state
// ------------------------------------------------------------------
let connectedPlayers = 0;

const gameState = {
  paddles : {
    left  : { y: 50, score: 0 },
    right : { y: 50, score: 0 }
  },
  ball: {
    x: 400,
    y: 300,
    dx: 0,
    dy: 0,
    radius: BALL_RAD
  },
  blocks: [],
  reward: 0
};

// ------------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------------
function initBlocks() {
  const cols = 10;
  const rows = 12;
  const startX = (CANVAS_W - (cols * (BLOCK_W + BRICK_GAP) - BRICK_GAP)) / 2;

  gameState.blocks = [];                // clear existing
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      gameState.blocks.push({
        x      : startX + col * (BLOCK_W + BRICK_GAP),
        y      : 40 + row * (BLOCK_H + BRICK_GAP),
        active : true,
        color  : col < 5 ? 'red' : 'purple'         // score to left / right
      });
    }
  }
}

function circleRectCollides(ball, block) {
  // Axis-aligned bounding-box test (good enough for our small ball)
  return (
    ball.x + BALL_RAD > block.x &&
    ball.x - BALL_RAD < block.x + BLOCK_W &&
    ball.y + BALL_RAD > block.y &&
    ball.y - BALL_RAD < block.y + BLOCK_H
  );
}

// ------------------------------------------------------------------
//  Game loop
// ------------------------------------------------------------------
function startGameLoop() {
  setInterval(() => {
    if (connectedPlayers === 0) return;

    // ---- 1) keep a copy of the PREVIOUS position
    const prevX = gameState.ball.x;
    const prevY = gameState.ball.y;

    // ---- 2) move the ball
    gameState.ball.x += gameState.ball.dx;
    gameState.ball.y += gameState.ball.dy;

    // ---- 3) wall collisions (top / bottom)
    if (gameState.ball.y - BALL_RAD < 0 || gameState.ball.y + BALL_RAD > CANVAS_H) {
      gameState.ball.dy *= -1;
    }

    // ---- 4) paddle collisions
    const leftPaddle  = gameState.paddles.left;
    const rightPaddle = gameState.paddles.right;

    if (
      gameState.ball.x - BALL_RAD < 20 &&
      gameState.ball.y > leftPaddle.y &&
      gameState.ball.y < leftPaddle.y + 100
    ) {
      gameState.ball.dx = Math.abs(gameState.ball.dx);
    }

    if (
      gameState.ball.x + BALL_RAD > CANVAS_W - 20 &&
      gameState.ball.y > rightPaddle.y &&
      gameState.ball.y < rightPaddle.y + 100
    ) {
      gameState.ball.dx = -Math.abs(gameState.ball.dx);
    }

    // ---- 5) brick collisions  (fixed)
    for (const block of gameState.blocks) {
      if (!block.active) continue;
      if (!circleRectCollides(gameState.ball, block)) continue;

      // deactivate brick & update score
      block.active = false;
      if (block.color === 'red')  leftPaddle.score++;
      else                        rightPaddle.score++;

      // Determine hit side using previous position
      const cameFromLeft   = prevX + BALL_RAD <= block.x;
      const cameFromRight  = prevX - BALL_RAD >= block.x + BLOCK_W;
      const cameFromAbove  = prevY + BALL_RAD <= block.y;
      const cameFromBelow  = prevY - BALL_RAD >= block.y + BLOCK_H;

      if (cameFromLeft || cameFromRight) {
        gameState.ball.dx *= -1;
        // push ball outside block
        gameState.ball.x = cameFromLeft
          ? block.x - BALL_RAD - 1
          : block.x + BLOCK_W + BALL_RAD + 1;
      } else if (cameFromAbove || cameFromBelow) {
        gameState.ball.dy *= -1;
        // push ball outside block
        gameState.ball.y = cameFromAbove
          ? block.y - BALL_RAD - 1
          : block.y + BLOCK_H + BALL_RAD + 1;
      } else {
        // corner case: flip both
        gameState.ball.dx *= -1;
        gameState.ball.dy *= -1;
      }

      break;               // ***** stop after FIRST brick hit *****
    }

      // Reset ball if out of bounds on left
      if (gameState.ball.x < 0) {
        gameState.ball.x = 400;
        gameState.ball.y = 300;
        gameState.ball.dx = -4;
        gameState.ball.dy = -4;
        io.emit('punish', gameState);
      }

      // Reset ball if out of bounds on right
      if (gameState.ball.x > 800) {
        gameState.ball.x = 400;
        gameState.ball.y = 300;
        gameState.ball.dx = -4;
        gameState.ball.dy = -4;
      }

    // ---- 7) broadcast
    if (true) io.emit('gameState', gameState);
  }, 1000 / FPS);
}

// ------------------------------------------------------------------
//  Socket.IO handlers
// ------------------------------------------------------------------
io.on('connection', (socket) => {
  connectedPlayers++;
  console.log(`Player connected (Total: ${connectedPlayers})`);

  // kick-off ball when first player joins
  if (connectedPlayers === 1) {
    Object.assign(gameState.ball, { x: 600, y: 300, dx: -4, dy: -4 });
  }

  socket.emit('gameState', gameState);

  socket.on('paddleMove', (data) => {
    try {
      if (data.side === 'left') {
        gameState.paddles.left.y  = Math.max(0, Math.min(500, data.y));
      } else if (data.side === 'right') {
        gameState.paddles.right.y = Math.max(0, Math.min(500, data.y));
      }
    } catch (err) {
      console.error(err);
    }
  });

  socket.on('reset', () => {
    initBlocks();
    gameState.paddles.left.score  = 0;
    gameState.paddles.right.score = 0;
    io.emit('gameState', gameState);
  });

  socket.on('reward', (data) => {
    try {
      gameState.reward = data.reward;
    } catch (err) {
      console.error(err);
    }
  })

  socket.on('disconnect', () => {
    connectedPlayers--;
    console.log(`Player disconnected (Remaining: ${connectedPlayers})`);

    if (connectedPlayers === 0) {
      Object.assign(gameState.ball, { x: 600, y: 300, dx: 0, dy: 0 });
      gameState.paddles.left.score  = 0;
      gameState.paddles.right.score = 0;
      initBlocks();
    }
  });
});

// ------------------------------------------------------------------
initBlocks();
startGameLoop();

server.listen(3000, () => {
  console.log('Server running on port 3000');
});
