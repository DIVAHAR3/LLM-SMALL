const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const API_KEY = import.meta.env.VITE_API_KEY || ''

// NOTE: VITE_API_KEY ends up inside the built JS bundle -- anyone loading
// this page can read it. That's an accepted simplification only because
// this project is local-only and single-user right now (see docs/SECURITY.md).
// A real public deployment needs per-user auth or a backend-for-frontend
// proxy instead, not a static key baked into client code.

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
    const body = await response.json().catch(() => ({}))
    const detail =
      typeof body.detail === 'string'
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail.map((e) => e.msg).join('; ')
          : response.statusText
    throw new Error(detail)
  }

  const data = await response.json()
  return data.text
}
