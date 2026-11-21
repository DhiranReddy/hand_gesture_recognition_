/**
 * Signal — Service Worker for offline capability & PWA caching
 */

const CACHE_NAME = "signal-pwa-v1";

const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./presenter.html",
  "./404.html",
  "./styles.css?v=13",
  "./manifest.webmanifest",
  "./public/model/words.json",
  "./src/audio.js",
  "./src/calibration.js",
  "./src/classifier.js",
  "./src/features.js",
  "./src/geometry.js",
  "./src/guide.js",
  "./src/handshape.js",
  "./src/main.js?v=13",
  "./src/recognizer.js",
  "./src/rules.js",
  "./src/speech.js",
  "./src/words.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.map((key) => {
            if (key !== CACHE_NAME) {
              return caches.delete(key);
            }
          }),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only handle GET requests and http/https schemes
  if (request.method !== "GET" || !request.url.startsWith("http")) {
    return;
  }

  // Handle same-origin assets with cache-first, network fallback
  const url = new URL(request.url);
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        if (cachedResponse) {
          // Revalidate in background for HTML/JS
          fetch(request)
            .then((networkResponse) => {
              if (networkResponse && networkResponse.ok) {
                caches.open(CACHE_NAME).then((cache) => cache.put(request, networkResponse));
              }
            })
            .catch(() => {});
          return cachedResponse;
        }

        return fetch(request).then((networkResponse) => {
          if (networkResponse && networkResponse.ok) {
            const clone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return networkResponse;
        });
      }),
    );
    return;
  }

  // Handle CDN / external assets (fonts, MediaPipe) with network-first, fallback to cache
  event.respondWith(
    fetch(request)
      .then((networkResponse) => {
        if (networkResponse && networkResponse.ok) {
          const clone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        }
        return networkResponse;
      })
      .catch(() => caches.match(request)),
  );
});
