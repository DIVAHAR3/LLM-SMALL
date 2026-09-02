import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'

vi.mock('./api')

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks() // restoreAllMocks doesn't reset call counts for vi.mock()-auto-mocked exports
  })

  it('shows the empty state initially and a disabled send button', () => {
    render(<App />)
    expect(screen.getByText(/send a prompt/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
  })

  it('sends a prompt and renders the assistant response', async () => {
    api.generateText.mockResolvedValue('a generated reply')
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hello there')
    await user.click(screen.getByRole('button', { name: /send/i }))

    expect(screen.getByText('hello there')).toBeInTheDocument() // user bubble appears immediately
    await waitFor(() => expect(screen.getByText('a generated reply')).toBeInTheDocument())
    expect(api.generateText).toHaveBeenCalledWith('hello there', expect.any(Object))
  })

  it('shows a loading state while waiting, then clears it', async () => {
    let resolvePromise
    api.generateText.mockReturnValue(
      new Promise((resolve) => {
        resolvePromise = resolve
      }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))

    expect(screen.getByText(/generating/i)).toBeInTheDocument()
    resolvePromise('done')
    await waitFor(() => expect(screen.queryByText(/generating/i)).not.toBeInTheDocument())
  })

  it('shows an error banner when the API call fails', async () => {
    api.generateText.mockRejectedValue(new Error('Rate limit exceeded. Try again later.'))
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => expect(screen.getByText(/rate limit exceeded/i)).toBeInTheDocument())
  })

  it('clear button empties the message history and error state', async () => {
    api.generateText.mockResolvedValue('reply')
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => expect(screen.getByText('reply')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /clear/i }))

    expect(screen.queryByText('hi')).not.toBeInTheDocument()
    expect(screen.queryByText('reply')).not.toBeInTheDocument()
    expect(screen.getByText(/send a prompt/i)).toBeInTheDocument()
  })

  it('does not send while a request is already in flight', async () => {
    let resolvePromise
    api.generateText.mockReturnValue(
      new Promise((resolve) => {
        resolvePromise = resolve
      }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))
    expect(screen.getByPlaceholderText(/type a prompt/i)).toBeDisabled()

    resolvePromise('done')
    await waitFor(() => expect(api.generateText).toHaveBeenCalledTimes(1))
  })
})
