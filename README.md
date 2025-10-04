# RAG Project Management Backend

This repository contains a minimal Node.js backend server for the RAG Project Management workspace. The server is built with [Express](https://expressjs.com/) and exposes a couple of basic routes to verify that the service is running.

## Prerequisites

- [Node.js](https://nodejs.org/) (version 18 or newer is recommended)
- [npm](https://www.npmjs.com/) or [pnpm](https://pnpm.io/) for package management

## Getting started

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server with automatic reloads:

   ```bash
   npm run dev
   ```

   This uses `nodemon` to watch the source files and restart the server on changes.

3. Or run the production server:

   ```bash
   npm start
   ```

## Available endpoints

- `GET /` – returns a welcome message.
- `GET /health` – returns a simple health check payload.

Both endpoints respond with JSON.

## Environment variables

- `PORT` – port number the server should listen on. Defaults to `3000` if not provided.

## Project structure

```
.
├── package.json
├── README.md
└── src
    └── server.js
```

Feel free to expand this starter project to include additional routes, middleware, and integrations required for your RAG project management needs.
