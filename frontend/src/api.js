const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const API_KEY = import.meta.env.VITE_API_KEY || ''

// NOTE: VITE_API_KEY ends up inside the built JS bundle -- anyone loading
// this page can read it. That's an accepted simplification only because
// this project is local-only and single-user right now (see docs/SECURITY.md).
// A real public deployment needs per-user auth or a backend-for-frontend
// proxy instead, not a static key baked into client code.

async function extractErrorDetail(response) {
  const body = await response.json().catch(() => ({}))
  if (typeof body.detail === 'string') return body.detail
  if (Array.isArray(body.detail)) return body.detail.map((e) => e.msg).join('; ')
  return response.statusText
}

export async function generateText(prompt, options = {}) {
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({ prompt, ...options }),
  })

  if (!response.ok) {
    throw new Error(await extractErrorDetail(response))
  }

  const data = await response.json()
  return data.text
}

// EventSource (the browser's built-in SSE client) only supports GET with no
// custom headers/body, but we need POST (to send the prompt) and a custom
// X-API-Key header -- so this reads the stream manually via fetch()'s
// ReadableStream body instead.
export async function generateTextStream(prompt, options = {}, onChunk) {
  const response = await fetch(`${API_URL}/generate/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({ prompt, ...options }),
  })

  if (!response.ok) {
    throw new Error(await extractErrorDetail(response))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })

    // SSE events are separated by a blank line; keep any trailing partial
    // event (the stream can split a chunk mid-event) in the buffer.
    const events = buffer.split('\n\n')
    buffer = events.pop()

    for (const event of events) {
      if (!event.startsWith('data: ')) continue
      const payload = JSON.parse(event.slice('data: '.length))
      if (payload.error) throw new Error(payload.error)
      if (payload.done) return
      onChunk(payload.chunk)
    }
  }
}

// Classical, deterministic image analysis (dimensions, colors, brightness,
// EXIF) -- no model, no training, no external AI call. See
// docs/IMAGE_ANALYSIS.md. Deliberately no Content-Type header: the browser
// sets multipart/form-data with the correct boundary itself for FormData
// bodies, and overriding it manually breaks the boundary.
export async function analyzeImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/analyze/image`, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
    },
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await extractErrorDetail(response))
  }

  return response.json()
}
