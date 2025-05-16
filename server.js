const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { json } = require('stream/consumers');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static('public'));

const FPS = 60;
const BLOCK_WIDTH = 50;
const BLOCK_HEIGHT = 40;
let connectedPlayers = 0;

const gameState = {
  paddles: {
    left: { y: 50, score: 0 },
    right: { y: 50, score: 0 }
  },
  ball: {
    x: 600,
    y: 300,
    dx: 0,
    dy: 0,
    radius: 8
  },
  blocks: []
};

function initBlocks() {
  const cols = 10;
  const rows = 12;
  const horizontalGap = (800 - (cols * (BLOCK_WIDTH + 5))) / 2;

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      gameState.blocks.push({
        x: horizontalGap + j * (BLOCK_WIDTH + 5),
        y: 40 + i * (BLOCK_HEIGHT + 5),
        active: true,
        color: j < 5 ? 'red' : 'purple' // Color based on column
      });
    }
  }
}

function startGameLoop() {
  let stateChanged = false;
  setInterval(() => {
    if (connectedPlayers > 0) {
      gameState.ball.x += gameState.ball.dx;
      gameState.ball.y += gameState.ball.dy;

      // Wall collisions (top/bottom)
      if (gameState.ball.y < 0 || gameState.ball.y > 600) {
        gameState.ball.dy *= -1;
        stateChanged = true;
        // console.log("first change")
      }

      // Paddle collisions
      if (gameState.ball.x - gameState.ball.radius < 20 &&
          gameState.ball.y > gameState.paddles.left.y &&
          gameState.ball.y < gameState.paddles.left.y + 100) {
        gameState.ball.dx = Math.abs(gameState.ball.dx);
        stateChanged = true;
        // console.log("second change")
      }

      if (gameState.ball.x + gameState.ball.radius > 800 - 20 &&
          gameState.ball.y > gameState.paddles.right.y &&
          gameState.ball.y < gameState.paddles.right.y + 100) {
        gameState.ball.dx = -Math.abs(gameState.ball.dx);
        stateChanged = true;
        // console.log("third change")
      }

      // Block collisions
      gameState.blocks.forEach(block => {
        if (block.active) {
          if (gameState.ball.x + gameState.ball.radius > block.x &&
              gameState.ball.x - gameState.ball.radius < block.x + BLOCK_WIDTH &&
              gameState.ball.y + gameState.ball.radius > block.y &&
              gameState.ball.y - gameState.ball.radius < block.y + BLOCK_HEIGHT) {
            
            block.active = false;
            stateChanged = true;
            // console.log("fourth change")
            
            // Update scores based on block color
            if (block.color === 'red') {
              gameState.paddles.left.score++;
            } else {
              gameState.paddles.right.score++;
            }

            // Collision direction detection
            const ballLeft = gameState.ball.x - gameState.ball.radius;
            const ballRight = gameState.ball.x + gameState.ball.radius;
            const ballTop = gameState.ball.y - gameState.ball.radius;
            const ballBottom = gameState.ball.y + gameState.ball.radius;

            const blockLeft = block.x;
            const blockRight = block.x + BLOCK_WIDTH;
            const blockTop = block.y;
            const blockBottom = block.y + BLOCK_HEIGHT;

            const overlaps = {
              left: ballRight - blockLeft,
              right: blockRight - ballLeft,
              top: ballBottom - blockTop,
              bottom: blockBottom - ballTop
            };

            const minOverlap = Math.min(...Object.values(overlaps));
            
            if (minOverlap === overlaps.left || minOverlap === overlaps.right) {
              gameState.ball.dx *= -1;
            } else {
              gameState.ball.dy *= -1;
            }
          }
        }
      });

      // Reset ball if out of bounds
      if (gameState.ball.x < 0 || gameState.ball.x > 800) {
        gameState.ball.x = 600;
        gameState.ball.y = 300;
        gameState.ball.dx = -4;
        gameState.ball.dy = -4;
      }
    }

    if (stateChanged) {
      io.emit('gameState', gameState);
    }
  }, 1000 / FPS);
}

io.on('connection', (socket) => {
  connectedPlayers++;
  console.log(`Player connected (Total: ${connectedPlayers})`);

  if (connectedPlayers === 1) {
    gameState.ball = {
      x: 600,
      y: 300,
      dx: -4,
      dy: -4,
      radius: 8
    };
  }

  socket.emit('gameState', gameState);
  
  socket.onAny((data) => {
    console.log(data)
    // const jsonObj = JSON.parse(data.JSON);
    // console.log(jsonObj)
  });

  socket.on('paddleMove', (data) => {
    console.log(data)
    try {
      // data = JSON.parse(data)
      
      if (data.side == 'left') {
        gameState.paddles.left.y = Math.max(0, Math.min(500, data.y));
      }
      if (data.side == 'right') {
        gameState.paddles.right.y = Math.max(0, Math.min(500, data.y));
      }
    } catch (error) {
      console.error();
    }
    
  });

  socket.on('disconnect', () => {
    connectedPlayers--;
    console.log(`Player disconnected (Remaining: ${connectedPlayers})`);
    
    if (connectedPlayers === 0) {
      gameState.ball = {
        x: 600,
        y: 300,
        dx: 0,
        dy: 0,
        radius: 8
      };
      // Reset scores when all players leave
      gameState.paddles.left.score = 0;
      gameState.paddles.right.score = 0;
      initBlocks(); // Reset blocks
    }
  });
});

initBlocks();
startGameLoop();

server.listen(3000, () => {
  console.log('Server running on port 3000');
});