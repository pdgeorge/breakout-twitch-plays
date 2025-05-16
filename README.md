# Twitch Plays Breakout Game

A real-time multiplayer Breakout game where two players compete to destroy blocks. Features synchronized gameplay, colored blocks, and score tracking.

## Features
- 🕹️ Two-player controls (left/right paddles)
- 🎨 Color-coded blocks (red left side, purple right side)
- 📊 Real-time score tracking
- 🏓 Ball physics with precise collision detection
- 🌐 WebSocket-based synchronization
- 🖱️ Mouse-controlled paddles

## Data objects for websocket communication

### Server sending to client
```
{ 
    paddles: {
        left: {
            y: 123,
            score: 456
        },
        right: {
            y: 123,
            score: 456
        }
    },
    ball: {
        x: 1,
        y: 1,
        dx: -1,
        dy: 1,
        radius: 8
    },
    blocks: [
        { x: 125, y: 40, active: true, color: 'red' },
        { x: 180, y: 40, active: false, color: 'red' },
        etc.
    ]
}
```

### Client sending to server
```
{
    side: 'right',
    y: 123
}
```

## Installation

### Prerequisites
- Node.js (v16+)
- npm (comes with Node.js)

### Setup Steps
1. **Clone the repository**

2. **Install socket.**
    `npm install express socket.io`

3. **Start the server.**
    `node server.js`
