# RAG Project Management Backend

This repository contains a minimal Node.js backend server for the RAG Project Management workspace. The server is built with [Express](https://expressjs.com/) and now includes a lightweight Retrieval Augmented Generation (RAG) pipeline that can ingest Excel sprint backlogs and answer natural language questions about the imported tasks.

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
- `POST /ingest` – accepts an Excel `.xls`/`.xlsx` file (field name `file`), parses the first worksheet using the sprint backlog template, and stores the rows in the in-memory knowledge base.
- `POST /query` – accepts a JSON payload containing a `question` string (and optional `topK` number) and returns the most relevant tasks alongside a synthesized answer generated from the best match.

Example query payload:

```bash
curl -X POST http://localhost:3000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Which tasks are assigned to QA_Olivia?",
    "topK": 2
  }'
```

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
