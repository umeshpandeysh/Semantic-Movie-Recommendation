/**
 * recommendations.js
 * Semantic Movie Recommendation Engine — Recommendation API Route
 *
 * POST /api/recommend  → Returns top-K semantically similar movies
 * GET  /api/movies     → Returns movie list
 */

const express = require('express');
const router = express.Router();
const { execFile } = require('child_process');
const path = require('path');

/**
 * POST /api/recommend
 * Body: { query: string, top_k: number }
 * Returns: { recommendations: [{ title, genre, similarity_score, overview }] }
 */
router.post('/recommend', (req, res) => {
  const { query, top_k = 10 } = req.body;

  if (!query || typeof query !== 'string' || query.trim().length === 0) {
    return res.status(400).json({ error: 'Query text is required.' });
  }

  // Call Python embedding service
  const scriptPath = path.join(__dirname, '..', 'services', 'embedding_service.py');

  execFile('python', [scriptPath, '--query', query.trim(), '--top_k', String(top_k)], (err, stdout, stderr) => {
    if (err) {
      console.error('[ERROR] Embedding service failed:', stderr);
      return res.status(500).json({
        error: 'Recommendation service unavailable.',
        detail: stderr
      });
    }

    try {
      const results = JSON.parse(stdout);
      res.json({ query: query.trim(), top_k, recommendations: results });
    } catch (parseErr) {
      console.error('[ERROR] Failed to parse embedding service output:', parseErr);
      res.status(500).json({ error: 'Failed to parse recommendation results.' });
    }
  });
});

/**
 * GET /api/movies
 * Returns the full list of movies in the corpus.
 */
router.get('/movies', (req, res) => {
  // Placeholder — replace with actual data loading
  res.json({
    message: 'Movie corpus endpoint. Add your movie metadata CSV to the embeddings/ folder.',
    total: 0,
    movies: []
  });
});

module.exports = router;
