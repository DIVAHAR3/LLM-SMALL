import { beforeEach, describe, expect, it, vi } from 'vitest'
import { generateText } from './api'

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
