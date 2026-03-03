'use strict';

const express = require('express');
const fetch = require('node-fetch');
const cheerio = require('cheerio');
const path = require('path');
const crypto = require('crypto');

const app = express();
app.use(express.json());
const PORT = process.env.PORT || 3500;

const FINNHUB_API_KEY    = process.env.FINNHUB_API_KEY || '';
const FINNHUB_CATEGORY   = process.env.FINNHUB_CATEGORY || 'general';
const VLLM_BASE_URL      = (process.env.VLLM_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const VLLM_MODEL         = process.env.VLLM_MODEL    || 'qwen3-32b-instruct-nvfp4';
const SCRAPE_MAX         = parseInt(process.env.SCRAPE_MAX_CHARS || '12000', 10);
const SCRAPE_CONCURRENCY = parseInt(process.env.SCRAPE_CONCURRENCY || '5', 10);

// allow tuning of LLM request timeout and retry behaviour via environment
const LLM_FETCH_TIMEOUT_MS = parseInt(process.env.LLM_FETCH_TIMEOUT_MS || '120000', 10); // 2min default
const LLM_RETRY_DELAY_MS   = parseInt(process.env.LLM_RETRY_DELAY_MS   || '10000', 10); // 10s between retries

console.log(`[vllm] base=${VLLM_BASE_URL}  model=${VLLM_MODEL}`);
if (!FINNHUB_API_KEY) console.warn('[feed] FINNHUB_API_KEY not set — requests will be unauthenticated and rate-limited');


// ── ticker extraction helper ───────────────────────────────────────────────
// simple regex-based search for $SYMBOL or (SYMBOL)
function extractTickers(text) {
  if (!text) return [];
  const found = new Set();
  let m;
  const re1 = /\$([A-Z]{1,5})\b/g;
  while ((m = re1.exec(text)) !== null) found.add(m[1]);
  const re2 = /\(([A-Z]{1,5})\)/g;
  while ((m = re2.exec(text)) !== null) found.add(m[1]);
  return Array.from(found);
}

const SUMMARY_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
const summaryCache = new Map(); // url -> { ts, text }

// ── Build graph JSON ──────────────────────────────────────────────────────────
// Items are pre-enriched Finnhub articles: { headline, url, summary, datetime, tickers[] }
// Tickers are extracted from scraped article text before this function is called.
function buildGraph(rawItems) {
  const nodes = [];
  const links = [];
  const tickerMap  = {};  // ticker symbol -> node id
  const tickerCount= {};

  rawItems.forEach((item, i) => {
    const title   = item.headline || `Article ${i}`;
    const link    = item.url || '';
    // Use a stable hash of the article URL (falling back to title) so that
    // the same article always gets the same node ID across feed fetches.
    const rawId   = link || title || String(i);
    const id      = `article-${crypto.createHash('md5').update(rawId).digest('hex').slice(0, 10)}`;
    const desc    = item.summary || '';
    const pubDate = item.datetime ? new Date(item.datetime * 1000).toUTCString() : '';
    const tickers = Array.isArray(item.tickers) ? item.tickers : [];

    nodes.push({ id, label: title, type: 'article', url: link, description: desc, pubDate });

    // ── Ticker nodes only
    tickers.forEach(tkr => {
      const key = tkr.toUpperCase().trim();
      if (!key) return;
      if (!tickerMap[key]) {
        tickerMap[key] = `ticker-${key}`;
        tickerCount[key] = 0;
        nodes.push({ id: tickerMap[key], label: key, type: 'ticker' });
      }
      tickerCount[key]++;
      links.push({ source: id, target: tickerMap[key] });
    });
  });

  // Embed connection counts on ticker nodes
  nodes.forEach(n => {
    if (n.type === 'ticker') n.connections = tickerCount[n.label] || 1;
  });

  return { nodes, links };
}

// ── Concurrency-limited URL scraper ─────────────────────────────────────────
// Scrapes up to `limit` article URLs in parallel; returns an array of scraped
// text strings (empty string on failure) in the same order as `items`.
async function scrapeArticles(items, limit) {
  const results = new Array(items.length).fill('');
  let idx = 0;

  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      const url = items[i].url;
      if (!url) continue;
      try {
        results[i] = await fetchArticleText(url);
      } catch (e) {
        console.warn(`[scrape] ${url}: ${e.message}`);
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

// ── Shared feed/graph logic ─────────────────────────────────────────────────
// Returns { graph, feedHash }
async function buildGraphFromFeed() {
  let apiUrl = `https://finnhub.io/api/v1/news?category=${FINNHUB_CATEGORY}&token=${FINNHUB_API_KEY}`;
  if (cachedLatestId) apiUrl += `&minId=${cachedLatestId}`;
  const feedResp = await fetch(apiUrl, {
    headers: { 'User-Agent': 'GraphVis/1.0 (news graph visualizer)' },
    timeout: 10000,
  });
  if (!feedResp.ok) throw new Error(`Finnhub returned ${feedResp.status}`);

  const articles = await feedResp.json();
  if (!Array.isArray(articles)) throw new Error('Unexpected Finnhub response structure');

  // Update cachedLatestId to the highest numeric article id we've seen so far
  if (articles.length) {
    const maxId = articles.reduce((m, a) => {
      const n = Number(a.id) || 0;
      return Math.max(m, n);
    }, 0);
    if (maxId > (Number(cachedLatestId) || 0)) cachedLatestId = String(maxId);
  }

  // Stable hash based on article IDs — only triggers re-scrape when feed changes.
  const feedHash = crypto.createHash('md5').update(articles.map(a => a.id).join(',')).digest('hex');

  // Scrape every article URL concurrently (bounded) and extract tickers from full text.
  console.log(`[feed] ${articles.length} articles from Finnhub, scraping URLs (concurrency=${SCRAPE_CONCURRENCY})…`);
  const scrapedTexts = await scrapeArticles(articles, SCRAPE_CONCURRENCY);

  const enriched = articles.map((article, i) => {
    const combined = [article.headline, article.summary, scrapedTexts[i]].filter(Boolean).join(' ');
    const tickers  = extractTickers(combined);
    // Finnhub populates `related` with a ticker symbol for company-specific news.
    if (article.related) {
      article.related.split(/[,\s]+/)
        .map(t => t.trim().toUpperCase())
        .filter(t => /^[A-Z]{1,5}$/.test(t))
        .forEach(t => { if (!tickers.includes(t)) tickers.push(t); });
    }
    return { ...article, tickers };
  });

  const graph = buildGraph(enriched);
  console.log(`[graph] ${graph.nodes.length} nodes, ${graph.links.length} links`);
  return { graph, feedHash };
}

// ── /api/graph  ───────────────────────────────────────────────────────────────
app.get('/api/graph', async (req, res) => {
  try {
    // return the current accumulated graph if we have one, otherwise
    // fetch a fresh copy.  This keeps the API consistent when polling
    // is active but the server has not yet started.
    if (cachedGraph) {
      res.json(cachedGraph);
    } else {
      const { graph } = await buildGraphFromFeed();
      res.json(graph);
    }
  } catch (err) {
    console.error('[graph] feed error:', err.message);
      // always propagate the error; stub graphs are no longer served.
    res.status(502).json({ error: 'feed fetch error', detail: err.message });
  }
});

// ── Live graph streaming support (poll + SSE) ───────────────────────────
const POLL_INTERVAL_MS = parseInt(process.env.POLL_INTERVAL_MS || '60000', 10);
let cachedGraph = null;
let cachedFeedHash = null;
let cachedLatestId = null; // use Finnhub `minId` to request only newer items
const sseClients = new Set();

function broadcastGraph(graph, feedHash) {
  const payload = JSON.stringify({ feedHash, graph });
  // send to each connected SSE client
  for (const client of sseClients) {
    try {
      client.write(`data: ${payload}\n\n`);
    } catch (e) {
      // ignore any write errors, client will be removed on close
    }
  }
}

app.get('/api/graph/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  // immediately send current graph if we have it
  if (cachedGraph) {
    res.write(`data: ${JSON.stringify({ feedHash: cachedFeedHash, graph: cachedGraph })}\n\n`);
  }

  sseClients.add(res);
  req.on('close', () => {
    sseClients.delete(res);
  });
});

// When the feed changes we merge new nodes/links into the existing
// cached graph rather than completely overwriting it.  This lets the
// visualization accumulate articles over time even if the RSS provider
// rotates older entries out of the current feed.
// merge newGraph into oldGraph; return true if anything was added
function mergeGraph(oldGraph, newGraph) {
  let added = false;

  const nodeIds = new Set(oldGraph.nodes.map(n => n.id));
  newGraph.nodes.forEach(n => {
    if (!nodeIds.has(n.id)) {
      oldGraph.nodes.push(n);
      nodeIds.add(n.id);
      added = true;
    }
  });

  // Build a lookup so we can update ticker connection counts when new
  // links targeting existing ticker nodes are merged in.
  const tickerNodeById = {};
  oldGraph.nodes.forEach(n => {
    if (n.type === 'ticker') tickerNodeById[n.id] = n;
  });

  const linkSet = new Set(oldGraph.links.map(l => `${l.source}->${l.target}`));
  newGraph.links.forEach(l => {
    const key = `${l.source}->${l.target}`;
    if (!linkSet.has(key)) {
      oldGraph.links.push(l);
      linkSet.add(key);
      added = true;
      // keep the connection count on existing ticker nodes in sync
      if (tickerNodeById[l.target]) {
        tickerNodeById[l.target].connections = (tickerNodeById[l.target].connections || 1) + 1;
      }
    }
  });

  return added;
}

async function pollFeed() {
  try {
    const { graph, feedHash } = await buildGraphFromFeed();

    if (!cachedGraph) {
      // first run - just adopt whatever we fetched
      cachedGraph = graph;
      cachedFeedHash = feedHash;
      console.log('[poll] initial feed load, broadcasting');
      broadcastGraph(cachedGraph, cachedFeedHash);
      return;
    }

    if (feedHash === cachedFeedHash) {
      console.log('[poll] feed unchanged');
      return;
    }

    // feed changed; only broadcast if merging actually adds something
    console.log('[poll] feed changed, attempting merge');
    const changed = mergeGraph(cachedGraph, graph);
    if (changed) {
      cachedFeedHash = feedHash;
      console.log('[poll] merged new items, broadcasting');
      broadcastGraph(cachedGraph, cachedFeedHash);
    } else {
      console.log('[poll] feed changed but no new nodes/links to add');
    }
  } catch (e) {
    console.error('[poll] error fetching feed:', e.message);
  }
}

// ── startup readiness helpers ───────────────────────────────────────────
async function waitForVLLM() {
  const url = `${VLLM_BASE_URL}/v1/models`;
  // repeatedly check until the vLLM service responds.  there's no
  // fallback, so a temporary outage should stall startup rather than
  // allow the app to spin up with dummy data.
  while (true) {
    try {
      const r = await fetch(url, { timeout: 5000 });
      if (r.ok) {
        console.log('[startup] vLLM service is ready');
        return;
      }
    } catch (e) {
      // ignore and retry
    }
    console.log('[startup] vLLM not ready yet, retrying in 2s...');
    await new Promise(r => setTimeout(r, 2000));
  }
}

// start polling loop only after ensuring the LLM endpoint is reachable
(async () => {
  await waitForVLLM();
  pollFeed();
  setInterval(pollFeed, POLL_INTERVAL_MS);
})();

// ── Article text extraction (trafilatura-style, via cheerio) ─────────────────
async function fetchArticleText(url) {
  const resp = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (compatible; GraphVis/1.0)' },
    timeout: 15000,
    redirect: 'follow',
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching article`);
  const html = await resp.text();

  const $ = cheerio.load(html);

  // Drop boilerplate elements
  $('script, style, noscript, iframe, nav, header, footer, aside, form,\
    figure, figcaption, [role="navigation"], [role="banner"], [role="complementary"],\
    .cookie-banner, .ad, .advertisement, .promo, .related, .sidebar').remove();

  // Prefer semantic article containers
  const candidate =
    $('article').first().text() ||
    $('main').first().text()    ||
    $('[role="main"]').first().text() ||
    $('body').text();

  return candidate
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
    .slice(0, SCRAPE_MAX);
}

// ── POST /api/summarize  ──────────────────────────────────────────────────────
// Body: { url: string, title?: string }
// Response: SSE stream  →  data: {"delta": "..."}  …  data: [DONE]
app.post('/api/summarize', async (req, res) => {
  const { url, title } = req.body || {};
  if (!url) return res.status(400).json({ error: 'url required' });

  // ── SSE headers ──────────────────────────────────────────────────────────
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = (obj) => {
    res.write(`data: ${JSON.stringify(obj)}\n\n`);
    if (res.flush) res.flush();
  };

  try {
    // 0 ── Cache hit: stream stored summary immediately
    const cached = summaryCache.get(url);
    if (cached && (Date.now() - cached.ts) < SUMMARY_CACHE_TTL_MS) {
      console.log('[summarize] cache hit for', url);
      send({ status: 'cached' });
      if (cached.tickers) send({ tickers: cached.tickers });
      const CHUNK = 40;
      for (let i = 0; i < cached.text.length; i += CHUNK) {
        send({ delta: cached.text.slice(i, i + CHUNK) });
      }
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    // 1 ── Scrape article text
    send({ status: 'scraping' });
    let articleText;
    try {
      articleText = await fetchArticleText(url);
    } catch (e) {
      send({ status: 'scrape_failed', detail: e.message });
      articleText = title ? `Article title: ${title}` : 'Article text could not be retrieved.';
    }

    // extract tickers and inform client
    const tickers = extractTickers((title || '') + ' ' + articleText);
    if (tickers.length) send({ tickers });

    // 2 ── Call vLLM
    send({ status: 'summarizing' });

    const vllmResp = await fetch(`${VLLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      timeout: 60000,
      body: JSON.stringify({
        model: VLLM_MODEL,
        stream: true,
        max_tokens: 300,
        temperature: 0.3,
        messages: [
          {
            role: 'system',
            content:
              'You are a concise technology news analyst. ' +
              'Summarize the following article in 3-5 sentences. ' +
              'Focus on key facts, implications, and why it matters to the tech industry. ' +
              'Be direct and specific. Do not pad with filler phrases.',
          },
          {
            role: 'user',
            content:
              (title ? `Title: ${title}\n\n` : '') +
              `Article:\n${articleText}`,
          },
        ],
      }),
    });

    if (!vllmResp.ok) {
      const errText = await vllmResp.text();
      throw new Error(`vLLM ${vllmResp.status}: ${errText.slice(0, 200)}`);
    }

    // 3 ── Forward SSE tokens to client; accumulate for cache
    let fullText = '';
    const decoder = new TextDecoder();
    for await (const chunk of vllmResp.body) {
      const text = decoder.decode(chunk, { stream: true });
      for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') { res.write('data: [DONE]\n\n'); break; }
        try {
          const parsed = JSON.parse(payload);
          const delta  = parsed?.choices?.[0]?.delta?.content;
          if (delta) { send({ delta }); fullText += delta; }
        } catch { /* partial chunk – skip */ }
      }
    }

    // 4 ── Store in cache
    if (fullText) {
      summaryCache.set(url, { ts: Date.now(), text: fullText, tickers });
      console.log('[summarize] cached summary for', url, 'tickers=', tickers);
    }

  } catch (err) {
    console.error('[summarize] error:', err.message);
    send({ error: err.message });
  }

  res.end();
});

// ── Serve bundled modules from node_modules (avoids CORS/remote deps)
app.use('/modules', express.static(path.join(__dirname, 'node_modules')));

// ── Static frontend ───────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

app.listen(PORT, () => {
  console.log(`Graph-vis running → http://localhost:${PORT}`);
});
