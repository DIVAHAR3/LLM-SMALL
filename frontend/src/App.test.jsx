import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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

  it('sends a prompt and renders the response as it streams in', async () => {
    api.generateTextStream.mockImplementation(async (prompt, options, onChunk) => {
      onChunk('a ')
      onChunk('generated ')
      onChunk('reply')
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hello there')
    await user.click(screen.getByRole('button', { name: /send/i }))

    expect(screen.getByText('hello there')).toBeInTheDocument() // user bubble appears immediately
    await waitFor(() => expect(screen.getByText('a generated reply')).toBeInTheDocument())
    expect(api.generateTextStream).toHaveBeenCalledWith('hello there', expect.any(Object), expect.any(Function))
  })

  it('shows a waiting indicator before the first chunk, then the growing text', async () => {
    let deliverChunk
    let finishStream
    api.generateTextStream.mockImplementation(
      (prompt, options, onChunk) =>
        new Promise((resolve) => {
          deliverChunk = onChunk
          finishStream = resolve
        }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))

    expect(screen.getByText('…')).toBeInTheDocument()

    deliverChunk('partial')
    await waitFor(() => expect(screen.getByText('partial')).toBeInTheDocument())
    expect(screen.queryByText('…')).not.toBeInTheDocument()

    finishStream()
    await waitFor(() => expect(screen.getByPlaceholderText(/type a prompt/i)).not.toBeDisabled())
  })

  it('shows an error banner when the stream fails, keeping any partial text', async () => {
    api.generateTextStream.mockImplementation(async (prompt, options, onChunk) => {
      onChunk('partial output')
      throw new Error('Rate limit exceeded. Try again later.')
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => expect(screen.getByText(/rate limit exceeded/i)).toBeInTheDocument())
    expect(screen.getByText('partial output')).toBeInTheDocument()
  })

  it('drops the empty placeholder bubble when the stream fails before any chunk arrives', async () => {
    api.generateTextStream.mockRejectedValue(new Error('Invalid or missing API key.'))
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => expect(screen.getByText(/invalid or missing api key/i)).toBeInTheDocument())
    expect(container.querySelectorAll('.bubble.assistant')).toHaveLength(0)
  })

  it('clear button empties the message history and error state', async () => {
    api.generateTextStream.mockImplementation(async (prompt, options, onChunk) => onChunk('reply'))
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

  it('pastes an image, sends it for analysis, and renders the returned JSON', async () => {
    const analysis = { format: 'PNG', width: 10, height: 10, dominant_colors: [{ hex: '#000000', percent: 100 }] }
    api.analyzeImage.mockResolvedValue(analysis)
    render(<App />)

    const file = new File(['fake bytes'], 'photo.png', { type: 'image/png' })
    const clipboardData = { items: [{ type: 'image/png', getAsFile: () => file }] }
    fireEvent.paste(screen.getByPlaceholderText(/type a prompt/i), { clipboardData })

    expect(screen.getByText(/pasted image/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/"format": "PNG"/)).toBeInTheDocument())
    expect(api.analyzeImage).toHaveBeenCalledWith(file)
  })

  it('shows an error banner when image analysis fails', async () => {
    api.analyzeImage.mockRejectedValue(new Error('could not decode image: broken'))
    render(<App />)

    const file = new File(['bad bytes'], 'bad.png', { type: 'image/png' })
    const clipboardData = { items: [{ type: 'image/png', getAsFile: () => file }] }
    fireEvent.paste(screen.getByPlaceholderText(/type a prompt/i), { clipboardData })

    await waitFor(() => expect(screen.getByText(/could not decode image/i)).toBeInTheDocument())
  })

  it('ignores a plain text paste (no image clipboard item)', async () => {
    render(<App />)

    const clipboardData = { items: [{ type: 'text/plain', getAsFile: () => null }] }
    fireEvent.paste(screen.getByPlaceholderText(/type a prompt/i), { clipboardData })

    expect(api.analyzeImage).not.toHaveBeenCalled()
  })

  it('ignores a pasted image while a request is already in flight', async () => {
    let finishStream
    api.generateTextStream.mockReturnValue(
      new Promise((resolve) => {
        finishStream = resolve
      }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))
    expect(screen.getByPlaceholderText(/type a prompt/i)).toBeDisabled()

    const file = new File(['fake bytes'], 'photo.png', { type: 'image/png' })
    const clipboardData = { items: [{ type: 'image/png', getAsFile: () => file }] }
    fireEvent.paste(screen.getByPlaceholderText(/type a prompt/i), { clipboardData })

    expect(api.analyzeImage).not.toHaveBeenCalled()
    finishStream()
  })

  it('does not send while a request is already in flight', async () => {
    let finishStream
    api.generateTextStream.mockReturnValue(
      new Promise((resolve) => {
        finishStream = resolve
      }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByPlaceholderText(/type a prompt/i), 'hi')
    await user.click(screen.getByRole('button', { name: /send/i }))
    expect(screen.getByPlaceholderText(/type a prompt/i)).toBeDisabled()

    finishStream()
    await waitFor(() => expect(api.generateTextStream).toHaveBeenCalledTimes(1))
  })
})
