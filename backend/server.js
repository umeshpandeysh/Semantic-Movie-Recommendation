/**
 * server.js
 * Semantic Movie Recommendation Engine — Express API Server
 *
 * Serves recommendation results computed via Python embedding service.
 * Run with: node server.js
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const recommendationsRouter = require('./routes/recommendations');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api', recommendationsRouter);

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Semantic Movie Recommendation API',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// Root
app.get('/', (req, res) => {
  res.json({
    message: 'Semantic Movie Recommendation API',
    endpoints: [
      'POST /api/recommend — Get movie recommendations',
      'GET  /api/movies   — List all movies',
      'GET  /api/health   — Health check'
    ]
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`[INFO] Semantic Movie Recommendation API running on http://localhost:${PORT}`);
  console.log(`[INFO] Health check: http://localhost:${PORT}/api/health`);
});

module.exports = app;
