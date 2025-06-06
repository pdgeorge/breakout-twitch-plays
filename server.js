// server.js
// ------------------------------------------------------------------
//  Breakout-/-Pong hybrid server  (bounce now uses true incident angle)
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
const FPS           = 60;
const BLOCK_W       = 50;
const BLOCK_H       = 40;
const BALL_RAD      = 4;
const CANVAS_W      = 800;
const CANVAS_H      = 600;
const BRICK_GAP     = 5;

const PADDLE_H      = 100;
const PADDLE_W      = 10;
const MAX_BOUNCE_ANGLE = Math.PI / 3;   // 60°

/* ------------------------------------------------------------------ */
/*  Game state                                                        */
/* ------------------------------------------------------------------ */
let connectedPlayers = 0;

const gameState = {
  paddles : {
    left  : { y: 50, score: 0 },
    right : { y: 50, score: 0 }
  },
  ball: {
    x: 400, y: 300,
    dx: 0,  dy: 0,
    radius: BALL_RAD
  },
  blocks: [],
  reward: 0
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
function getRandomIntInclusive(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function initBlocks() {
  const cols = 10, rows = 12;
  const startX = (CANVAS_W - (cols*(BLOCK_W+BRICK_GAP)-BRICK_GAP))/2;
  gameState.blocks = [];

  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      gameState.blocks.push({
        x:startX + c*(BLOCK_W+BRICK_GAP),
        y:40    + r*(BLOCK_H+BRICK_GAP),
        active:true,
        color:c<5?'red':'purple'
      });
    }
  }
}

function circleRectCollides(ball,b){
  return ball.x+BALL_RAD>b.x   && ball.x-BALL_RAD<b.x+BLOCK_W &&
         ball.y+BALL_RAD>b.y   && ball.y-BALL_RAD<b.y+BLOCK_H;
}

function spawnBall(){
  gameState.ball.x = 400;
  gameState.ball.y = 400;
  gameState.ball.dx = getRandomIntInclusive(-4, 4);
  while (gameState.ball.dx === 0) gameState.ball.dx = getRandomIntInclusive(-4, 4);
  gameState.ball.dy = getRandomIntInclusive(-4, 4);
}

/**
 * Reflect the ball using its incident angle plus paddle-offset “english”.
 * Keeps the original speed magnitude.
 */
function applyPaddleBounce(paddleY, ball){
  // 1) distance from paddle-centre → [-1,1]
  const centre = paddleY + PADDLE_H/2;
  let   offset = (ball.y - centre)/(PADDLE_H/2);
  offset = Math.max(-1, Math.min(1, offset));

  // 2) incident & perfect-reflection angles
  const incident  = Math.atan2(ball.dy, ball.dx);      // -π … π
  const reflect   = Math.PI - incident;                // mirror over Y axis

  // 3) add english
  const newAngle  = reflect + offset*MAX_BOUNCE_ANGLE;

  // 4) preserve speed
  const speed = Math.hypot(ball.dx, ball.dy) || 4;
  ball.dx =  speed * Math.cos(newAngle);
  ball.dy =  speed * Math.sin(newAngle);
}

/* ------------------------------------------------------------------ */
/*  Main game loop                                                    */
/* ------------------------------------------------------------------ */
function startGameLoop(){
  setInterval(()=> {
    if (!connectedPlayers) return;

    const prevX=gameState.ball.x, prevY=gameState.ball.y;

    // move
    gameState.ball.x += gameState.ball.dx;
    gameState.ball.y += gameState.ball.dy;

    // top / bottom walls
    if (gameState.ball.y-BALL_RAD<0 || gameState.ball.y+BALL_RAD>CANVAS_H)
      gameState.ball.dy *= -1;

    const L = gameState.paddles.left,
          R = gameState.paddles.right;

    // ---------- LEFT paddle ----------
    if ( gameState.ball.dx < 0 &&                         // travelling left
         gameState.ball.x - BALL_RAD < PADDLE_W+10 &&
         gameState.ball.y > L.y && gameState.ball.y < L.y+PADDLE_H )
    {
      applyPaddleBounce(L.y, gameState.ball);
      io.emit('rewardHit', gameState);
    }

    // ---------- RIGHT paddle ----------
    if ( gameState.ball.dx > 0 &&                        // travelling right
         gameState.ball.x + BALL_RAD > CANVAS_W-(PADDLE_W+10) &&
         gameState.ball.y > R.y && gameState.ball.y < R.y+PADDLE_H )
    {
      applyPaddleBounce(R.y, gameState.ball);
    }

    // ---------- Brick collisions (unchanged) ----------
    for (const blk of gameState.blocks){
      if (!blk.active || !circleRectCollides(gameState.ball,blk)) continue;

      blk.active=false;
      (blk.color==='red'? L : R).score++;

      const cameL = prevX+BALL_RAD<=blk.x,
            cameR = prevX-BALL_RAD>=blk.x+BLOCK_W,
            cameU = prevY+BALL_RAD<=blk.y,
            cameD = prevY-BALL_RAD>=blk.y+BLOCK_H;

      if (cameL||cameR) gameState.ball.dx*=-1;
      else if (cameU||cameD) gameState.ball.dy*=-1;
      else { gameState.ball.dx*=-1; gameState.ball.dy*=-1; }
      break;
    }

    // ---------- Out of bounds ----------
    if (gameState.ball.x < 0){
      spawnBall();
      io.emit('punish',gameState);
    }
    if (gameState.ball.x > CANVAS_W) spawnBall();

    // ---------- Broadcast ----------
    io.emit('gameState', gameState);
  }, 1000/FPS);
}

/* ------------------------------------------------------------------ */
/*  Socket.IO handlers                                                */
/* ------------------------------------------------------------------ */
io.on('connection', (socket)=>{
  connectedPlayers++;
  console.log(`Player connected (${connectedPlayers})`);

  if (connectedPlayers===1)
    Object.assign(gameState.ball,{x:600,y:300,dx:-4,dy:-4});

  socket.emit('gameState',gameState);

  socket.on('paddleMove', ({side,y})=>{
    const clamp = v=> Math.max(0, Math.min(CANVAS_H-PADDLE_H, v));
    if(side==='left')  gameState.paddles.left.y  = clamp(y);
    if(side==='right') gameState.paddles.right.y = clamp(y);
  });

  socket.on('reset', ()=>{
    initBlocks();
    gameState.paddles.left.score = gameState.paddles.right.score = 0;
    io.emit('gameState', gameState);
  });

  socket.on('ballReset', ()=>{
    try {
    spawnBall();
    } catch (err) {
      console.error(err)
    }
  });

  socket.on('reward', d=> gameState.reward = d?.reward ?? 0);

  socket.on('disconnect', ()=>{
    connectedPlayers--;
    console.log(`Disconnect (${connectedPlayers})`);
    if (!connectedPlayers){
      Object.assign(gameState.ball,{x:600,y:300,dx:0,dy:0});
      gameState.paddles.left.score = gameState.paddles.right.score = 0;
      initBlocks();
    }
  });
});

/* ------------------------------------------------------------------ */
initBlocks();
startGameLoop();
server.listen(3000,()=> console.log('Server running on port 3000'));
