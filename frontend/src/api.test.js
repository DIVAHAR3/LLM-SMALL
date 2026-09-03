import { beforeEach, describe, expect, it, vi } from 'vitest'
import { analyzeImage, generateText, generateTextStream } from './api'

function makeMockReader(rawChunks) {
  let index = 0
  return {
    read: async () => {
      if (index >= rawChunks.length) return { done: true, value: undefined }
      const value = new TextEncoder().encode(rawChunks[index])
      index += 1
      return { done: false, value }
    },
  }
}

describe('generateText', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('POSTs to <API_URL>/generate with the API key header and returns the text', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ text: 'hello world' }) })

    const result = await generateText('hi', { max_new_tokens: 10 })

    expect(result).toBe('hello world')
    expect(global.fetch).toHaveBeenCalledTimes(1)
    const [url, options] = global.fetch.mock.calls[0]
    expect(url).toBe(`${import.meta.env.VITE_API_URL}/generate`)
    expect(options.method).toBe('POST')
    expect(options.headers['X-API-Key']).toBe(import.meta.env.VITE_API_KEY)
    expect(JSON.parse(options.body)).toEqual({ prompt: 'hi', max_new_tokens: 10 })
  })

  it('throws with the string detail message on a non-ok response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Unauthorized',
      json: async () => ({ detail: 'Invalid or missing API key.' }),
    })

    await expect(generateText('hi')).rejects.toThrow('Invalid or missing API key.')
  })

  it('joins pydantic validation-error arrays into a readable message', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Unprocessable Entity',
      json: async () => ({ detail: [{ msg: 'String should have at least 1 character' }] }),
    })

    await expect(generateText('')).rejects.toThrow('String should have at least 1 character')
  })

  it('falls back to statusText when the error body is not JSON', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Internal Server Error',
      json: async () => {
        throw new Error('not json')
      },
    })

    await expect(generateText('hi')).rejects.toThrow('Internal Server Error')
  })
})

describe('generateTextStream', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('POSTs to <API_URL>/generate/stream and calls onChunk for each SSE chunk in order', async () => {
    const reader = makeMockReader([
      'data: {"chunk": "a"}\n\n',
      'data: {"chunk": "b"}\n\n',
      'data: {"done": true}\n\n',
    ])
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => reader } })

    const received = []
    await generateTextStream('hi', { max_new_tokens: 10 }, (chunk) => received.push(chunk))

    expect(received).toEqual(['a', 'b'])
    const [url, options] = global.fetch.mock.calls[0]
    expect(url).toBe(`${import.meta.env.VITE_API_URL}/generate/stream`)
    expect(options.headers['X-API-Key']).toBe(import.meta.env.VITE_API_KEY)
    expect(JSON.parse(options.body)).toEqual({ prompt: 'hi', max_new_tokens: 10 })
  })

  it('reassembles an SSE event that arrives split across two reads', async () => {
    // the blank-line delimiter falls in the middle of what the network
    // layer actually delivers as two separate chunks -- must not lose data
    const reader = makeMockReader(['data: {"chunk": "hel', 'lo"}\n\n', 'data: {"done": true}\n\n'])
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => reader } })

    const received = []
    await generateTextStream('hi', {}, (chunk) => received.push(chunk))

    expect(received).toEqual(['hello'])
  })

  it('throws when the stream emits an error event, without calling onChunk for it', async () => {
    const reader = makeMockReader(['data: {"chunk": "a"}\n\n', 'data: {"error": "Rate limit exceeded."}\n\n'])
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => reader } })

    const received = []
    await expect(generateTextStream('hi', {}, (chunk) => received.push(chunk))).rejects.toThrow(
      'Rate limit exceeded.',
    )
    expect(received).toEqual(['a'])
  })

  it('throws with the response detail when the initial request itself fails', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Unauthorized',
      json: async () => ({ detail: 'Invalid or missing API key.' }),
    })

    await expect(generateTextStream('hi', {}, () => {})).rejects.toThrow('Invalid or missing API key.')
  })
})

describe('analyzeImage', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('POSTs the file as multipart form data with the API key header, no Content-Type override', async () => {
    const analysis = { format: 'PNG', width: 10, height: 10, dominant_colors: [] }
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => analysis })
    const file = new File(['fake image bytes'], 'photo.png', { type: 'image/png' })

    const result = await analyzeImage(file)

    expect(result).toEqual(analysis)
    expect(global.fetch).toHaveBeenCalledTimes(1)
    const [url, options] = global.fetch.mock.calls[0]
    expect(url).toBe(`${import.meta.env.VITE_API_URL}/analyze/image`)
    expect(options.method).toBe('POST')
    expect(options.headers['X-API-Key']).toBe(import.meta.env.VITE_API_KEY)
    expect(options.headers['Content-Type']).toBeUndefined()
    expect(options.body).toBeInstanceOf(FormData)
    expect(options.body.get('file')).toBe(file)
  })

  it('throws with the response detail on a non-ok response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'could not decode image: broken' }),
    })
    const file = new File(['not an image'], 'bad.png', { type: 'image/png' })

    await expect(analyzeImage(file)).rejects.toThrow('could not decode image: broken')
  })
})
